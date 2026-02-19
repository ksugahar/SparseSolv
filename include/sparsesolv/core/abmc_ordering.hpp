/**
 * @file abmc_ordering.hpp
 * @brief Algebraic Block Multi-Color (ABMC) ordering for parallel triangular solves
 *
 * Implements the ABMC ordering algorithm from:
 *   T. Iwashita, H. Nakashima, Y. Takahashi,
 *   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
 *    Sparse Triangular Solver in ICCG Method", IPDPS 2012
 *
 * Ported from JP-MARs/SparseSolv (MPL 2.0, original author: Shingo Hiruma).
 *
 * The algorithm works in two stages:
 * 1. Algebraic blocking: BFS aggregation groups nearby rows into blocks
 * 2. Multi-coloring: Greedy graph coloring of the block adjacency graph
 *
 * The resulting schedule enables parallel triangular solves:
 *   - Colors are processed sequentially (inter-color dependencies)
 *   - Blocks within the same color are processed in parallel (independent)
 *   - Rows within a block are processed sequentially
 */

#ifndef SPARSESOLV_CORE_ABMC_ORDERING_HPP
#define SPARSESOLV_CORE_ABMC_ORDERING_HPP

#include "types.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <map>
#include <cassert>

namespace sparsesolv {

/**
 * @brief ABMC schedule for parallel triangular solves
 *
 * Stores the two-level hierarchy (color -> block -> rows) and the
 * permutation arrays for reordering matrices and vectors.
 */
struct ABMCSchedule {
    /// color_list[c] = list of block indices assigned to color c
    std::vector<std::vector<index_t>> color_list;

    /// block_list[b] = list of row indices (in reordered space) in block b
    std::vector<std::vector<index_t>> block_list;

    /// ordering[old_row] = new_row
    std::vector<index_t> ordering;

    /// reverse_ordering[new_row] = old_row
    std::vector<index_t> reverse_ordering;

    /// Check if the schedule has been built
    bool is_built() const { return !color_list.empty(); }

    /// Number of colors
    index_t num_colors() const { return static_cast<index_t>(color_list.size()); }

    /// Number of blocks
    index_t num_blocks() const { return static_cast<index_t>(block_list.size()); }

    /**
     * @brief Build ABMC schedule from sparse matrix pattern
     *
     * @param row_ptr CSR row pointer array (size n+1)
     * @param col_idx CSR column index array
     * @param n Number of rows
     * @param block_size Target number of rows per block
     * @param target_colors Target number of colors for graph coloring
     */
    void build(const index_t* row_ptr, const index_t* col_idx, index_t n,
               int block_size, int target_colors) {
        if (n <= 0) {
            clear();
            return;
        }

        // Clamp parameters
        if (block_size < 1) block_size = 1;
        if (target_colors < 2) target_colors = 2;

        // Stage 1: BFS aggregation into blocks
        const index_t max_blocks = (n + block_size - 1) / block_size;
        std::vector<std::vector<index_t>> raw_block_list;
        std::vector<index_t> block_assign(n, -1);

        make_algebraic_blocks(row_ptr, col_idx, n, block_size,
                              max_blocks, raw_block_list, block_assign);

        const index_t actual_num_blocks = static_cast<index_t>(raw_block_list.size());

        // Stage 2: Build block-level adjacency graph (CSR)
        std::vector<index_t> blk_row_ptr;
        std::vector<index_t> blk_col_idx;
        build_block_graph(row_ptr, col_idx, n, actual_num_blocks,
                          block_assign, blk_row_ptr, blk_col_idx);

        // Stage 3: Multi-color the block graph
        std::vector<std::vector<index_t>> blk_color_list;
        std::vector<index_t> blk_ordering;
        std::vector<index_t> blk_reverse_ordering;

        color_graph(blk_row_ptr.data(), blk_col_idx.data(), actual_num_blocks,
                    target_colors, blk_color_list, blk_ordering, blk_reverse_ordering);

        // Stage 4: Build final row ordering from block ordering
        build_row_ordering(n, actual_num_blocks, raw_block_list,
                           blk_color_list, blk_reverse_ordering);
    }

    /// Clear all data
    void clear() {
        color_list.clear();
        block_list.clear();
        ordering.clear();
        reverse_ordering.clear();
    }

private:
    /**
     * @brief BFS-based algebraic block aggregation
     *
     * Groups nearby rows into blocks of approximately block_size rows each.
     * Uses breadth-first search on the adjacency graph of the matrix.
     *
     * @param row_ptr CSR row pointer
     * @param col_idx CSR column indices
     * @param n Matrix dimension
     * @param block_size Target rows per block
     * @param max_blocks Maximum number of blocks
     * @param raw_block_list Output: raw_block_list[b] = {original row indices}
     * @param block_assign Output: block_assign[row] = block id
     */
    static void make_algebraic_blocks(
        const index_t* row_ptr, const index_t* col_idx, index_t n,
        int block_size, index_t max_blocks,
        std::vector<std::vector<index_t>>& raw_block_list,
        std::vector<index_t>& block_assign)
    {
        raw_block_list.clear();
        raw_block_list.reserve(max_blocks);

        std::queue<index_t> bfs_queue;
        index_t next_unassigned = 0;

        for (index_t blk = 0; blk < max_blocks; ++blk) {
            // Find the first unassigned row
            while (next_unassigned < n && block_assign[next_unassigned] >= 0) {
                ++next_unassigned;
            }
            if (next_unassigned >= n) break;

            // Start a new block with this seed row
            raw_block_list.emplace_back();
            auto& current_block = raw_block_list.back();
            const index_t blk_id = static_cast<index_t>(raw_block_list.size() - 1);

            block_assign[next_unassigned] = blk_id;
            current_block.push_back(next_unassigned);

            // Enqueue neighbors of the seed
            for (index_t k = row_ptr[next_unassigned]; k < row_ptr[next_unassigned + 1]; ++k) {
                index_t j = col_idx[k];
                if (j != next_unassigned && block_assign[j] < 0) {
                    bfs_queue.push(j);
                }
            }

            // BFS expansion until block is full
            while (!bfs_queue.empty() &&
                   static_cast<int>(current_block.size()) < block_size) {
                index_t row = bfs_queue.front();
                bfs_queue.pop();

                if (block_assign[row] >= 0) continue;

                block_assign[row] = blk_id;
                current_block.push_back(row);

                if (static_cast<int>(current_block.size()) >= block_size) break;

                // Enqueue unassigned neighbors
                for (index_t k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
                    index_t j = col_idx[k];
                    if (block_assign[j] < 0) {
                        bfs_queue.push(j);
                    }
                }
            }

            // Clear queue for next block
            while (!bfs_queue.empty()) bfs_queue.pop();
        }

        // Handle any remaining unassigned rows (shouldn't happen normally)
        for (index_t i = 0; i < n; ++i) {
            if (block_assign[i] < 0) {
                if (raw_block_list.empty()) {
                    raw_block_list.emplace_back();
                }
                block_assign[i] = static_cast<index_t>(raw_block_list.size() - 1);
                raw_block_list.back().push_back(i);
            }
        }
    }

    /**
     * @brief Build block-level adjacency graph in CSR format
     *
     * Two blocks are adjacent if any row in one block has a nonzero entry
     * connecting to a row in the other block.
     */
    static void build_block_graph(
        const index_t* row_ptr, const index_t* col_idx, index_t n,
        index_t num_blocks, const std::vector<index_t>& block_assign,
        std::vector<index_t>& blk_row_ptr,
        std::vector<index_t>& blk_col_idx)
    {
        // Use set of neighbors per block to avoid duplicates
        std::vector<std::vector<index_t>> neighbors(num_blocks);

        for (index_t i = 0; i < n; ++i) {
            index_t bi = block_assign[i];
            for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                index_t j = col_idx[k];
                index_t bj = block_assign[j];
                if (bi != bj) {
                    neighbors[bi].push_back(bj);
                }
            }
        }

        // Sort and deduplicate each block's neighbor list
        for (auto& nbrs : neighbors) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }

        // Convert to CSR
        blk_row_ptr.resize(num_blocks + 1);
        blk_row_ptr[0] = 0;
        for (index_t b = 0; b < num_blocks; ++b) {
            blk_row_ptr[b + 1] = blk_row_ptr[b] + static_cast<index_t>(neighbors[b].size());
        }

        blk_col_idx.resize(blk_row_ptr[num_blocks]);
        for (index_t b = 0; b < num_blocks; ++b) {
            index_t pos = blk_row_ptr[b];
            for (index_t j : neighbors[b]) {
                blk_col_idx[pos++] = j;
            }
        }
    }

    /**
     * @brief Greedy multi-color ordering of a graph
     *
     * Assigns colors to nodes such that no two adjacent nodes in the
     * lower-triangular part share the same color. The coloring is greedy:
     * process nodes in order, cycling through candidate colors.
     *
     * @param row_ptr CSR row pointer of the graph
     * @param col_idx CSR column indices
     * @param n Number of nodes
     * @param target_colors Suggested number of colors
     * @param out_color_list Output: color_list[c] = {node indices with color c}
     * @param out_ordering Output: ordering[old] = new
     * @param out_reverse Output: reverse_ordering[new] = old
     */
    static void color_graph(
        const index_t* row_ptr, const index_t* col_idx, index_t n,
        int target_colors,
        std::vector<std::vector<index_t>>& out_color_list,
        std::vector<index_t>& out_ordering,
        std::vector<index_t>& out_reverse)
    {
        int num_colors = target_colors;

        // Determine minimum number of colors needed
        // (max lower-triangular degree + 1)
        for (index_t i = 0; i < n; ++i) {
            int lower_count = 0;
            for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                if (col_idx[k] < i) {
                    ++lower_count;
                }
            }
            if (lower_count >= num_colors) {
                num_colors = lower_count + 1;
            }
        }

        // Greedy coloring
        std::vector<index_t> node_color(n, -1);
        out_color_list.resize(num_colors);
        for (auto& cl : out_color_list) cl.clear();

        int candidate = 0;
        for (index_t i = 0; i < n; ++i) {
            // Try to assign candidate color, check for conflicts
            bool assigned = false;
            for (int trial = 0; trial < num_colors && !assigned; ++trial) {
                int test_color = (candidate + trial) % num_colors;
                bool conflict = false;

                // Check lower-triangular neighbors
                for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                    index_t j = col_idx[k];
                    if (j < i && node_color[j] == test_color) {
                        conflict = true;
                        break;
                    }
                }

                if (!conflict) {
                    node_color[i] = test_color;
                    out_color_list[test_color].push_back(i);
                    candidate = (test_color + 1) % num_colors;
                    assigned = true;
                }
            }

            // Fallback: if all colors conflict, create a new color
            if (!assigned) {
                node_color[i] = num_colors;
                out_color_list.emplace_back();
                out_color_list[num_colors].push_back(i);
                candidate = (num_colors + 1) % (num_colors + 1);
                ++num_colors;
            }
        }

        // Remove empty color groups
        out_color_list.erase(
            std::remove_if(out_color_list.begin(), out_color_list.end(),
                           [](const std::vector<index_t>& v) { return v.empty(); }),
            out_color_list.end());

        // Build ordering: enumerate nodes in color order
        out_ordering.resize(n);
        out_reverse.resize(n);
        index_t new_idx = 0;
        for (const auto& color_group : out_color_list) {
            for (index_t old_idx : color_group) {
                out_ordering[old_idx] = new_idx;
                out_reverse[new_idx] = old_idx;
                ++new_idx;
            }
        }
    }

    /**
     * @brief Build final row-level ordering from block ordering
     *
     * Enumerates rows in the order: color 0 blocks -> color 1 blocks -> ...
     * Within each block, rows are sorted by original index for locality.
     *
     * @note blk_color_list[c] contains ORIGINAL block indices with color c
     *       (as returned by color_graph). blk_reverse_ordering is not used.
     */
    void build_row_ordering(
        index_t n, index_t actual_num_blocks,
        const std::vector<std::vector<index_t>>& raw_block_list,
        const std::vector<std::vector<index_t>>& blk_color_list,
        const std::vector<index_t>& /*blk_reverse_ordering*/)
    {
        ordering.resize(n);
        reverse_ordering.resize(n);

        // Rebuild color_list and block_list in the new ordering
        color_list.resize(blk_color_list.size());
        block_list.resize(actual_num_blocks);

        index_t row_idx = 0;
        index_t global_blk_id = 0;
        for (index_t c = 0; c < static_cast<index_t>(blk_color_list.size()); ++c) {
            color_list[c].clear();
            for (index_t orig_blk : blk_color_list[c]) {
                // blk_color_list[c] already contains original block indices
                color_list[c].push_back(global_blk_id);

                // Sort rows within block for better locality
                auto rows = raw_block_list[orig_blk];
                std::sort(rows.begin(), rows.end());

                block_list[global_blk_id].clear();
                for (index_t orig_row : rows) {
                    ordering[orig_row] = row_idx;
                    reverse_ordering[row_idx] = orig_row;
                    block_list[global_blk_id].push_back(row_idx);
                    ++row_idx;
                }
                ++global_blk_id;
            }
        }

        assert(row_idx == n);
    }
};

} // namespace sparsesolv

#endif // SPARSESOLV_CORE_ABMC_ORDERING_HPP
