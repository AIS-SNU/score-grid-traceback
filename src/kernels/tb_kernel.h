#ifndef __TB_KERNEL__
#define __TB_KERNEL__

#define N_VALUE_T 'N'

#define GET_SUB_SCORE(subScore, query, target) \
	subScore = (query == target) ? _cudaMatchScore : -_cudaMismatchScore;\
	subScore = ((query == N_VALUE_T) || (target == N_VALUE_T)) ? 0 : subScore;\

__global__ void traceback_kernel_old(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *global_direction, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length) {
	int i, j;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int tb_matrix_size = maximum_sequence_length*maximum_sequence_length/8;

	i = device_res->target_batch_end[tid];
	j = device_res->query_batch_end[tid];

	int query_matched_idx = 0;
	int target_matched_idx = 0;

	while (i >= 0 && j >= 0) {
		int direction = (global_direction[tb_matrix_size*tid + maximum_sequence_length*(i>>3) + j] >> (28 - 4*(i & 7))) & 3;

		switch(direction) {
			case 0: // matched
			case 1: // mismatched
				result_query[maximum_sequence_length*tid + query_matched_idx++] = unpacked_query_batch[query_batch_offsets[tid] + j];
				result_target[maximum_sequence_length*tid + target_matched_idx++] = unpacked_target_batch[target_batch_offsets[tid] + i];
				i--;
				j--;
			break;
			case 2: // from upper cell
				result_query[maximum_sequence_length*tid + query_matched_idx++] = unpacked_query_batch[query_batch_offsets[tid] + j];
				result_target[maximum_sequence_length*tid + target_matched_idx++] = '-';
				j--;
			break;
			case 3: // left
				result_query[maximum_sequence_length*tid + query_matched_idx++] = '-';
				result_target[maximum_sequence_length*tid + target_matched_idx++] = unpacked_target_batch[target_batch_offsets[tid] + i];
				i--;
			break;
		}

	}
}

__global__ void traceback_kernel_dynamic(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, short2 *dblock_row, short2 *dblock_col, uint32_t *dp_matrix_offsets) {
	int i, j;
	int d_row, d_col;
	int inner_row, inner_col;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	i = device_res->target_batch_end[tid];
	j = device_res->query_batch_end[tid];

	d_row = i/DBLOCK_SIZE;
	d_col = j/DBLOCK_SIZE;
	inner_row = i%DBLOCK_SIZE;
	inner_col = j%DBLOCK_SIZE;

	extern __shared__ short3 sharedMemory[];
	short3 *scoreMatrixRow = sharedMemory+DBLOCK_SIZE*threadIdx.x;
	uint8_t directionMatrix[DBLOCK_SIZE*DBLOCK_SIZE];
	
	short3 tempLeftCell;
	short3 tempDiagCell;

	int query_matched_idx = 0;
	int target_matched_idx = 0;

	int x, y;
	int direction;
	short subScore, tmpScore;
	short3 left, up, diag;

	short3 entry;
	
	int dblock_offset = dp_matrix_offsets[tid];
	int dp_mtx_len = MAX(query_batch_lens[tid], target_batch_lens[tid]);


	while (d_row >= 0 && d_col >= 0) {
		// init first row of dynamic block
		tempDiagCell.x = d_col == 0 ? 0 : dblock_row[dblock_offset + dp_mtx_len*d_row + d_col*DBLOCK_SIZE - 1].y;
		tempDiagCell.y = d_row == 0 ? 0 : dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE - 1].y;
		tempDiagCell.z = d_row == 0 ? (d_col == 0 ? 0 : dblock_row[dblock_offset + d_col*DBLOCK_SIZE - 1].x) : (d_col == 0 ? dblock_col[dblock_offset + d_row*DBLOCK_SIZE - 1].x : dblock_row[dblock_offset + dp_mtx_len*d_row + d_col*DBLOCK_SIZE - 1].x);

		for (x = 0; x < DBLOCK_SIZE; x++) {
			if (d_row == 0) {
				scoreMatrixRow[x].x = 0;
				scoreMatrixRow[x].z = 0;
			} else {
				scoreMatrixRow[x].x = dblock_row[dblock_offset + dp_mtx_len*d_row + d_col*DBLOCK_SIZE + x].y;
				scoreMatrixRow[x].z = dblock_row[dblock_offset + dp_mtx_len*d_row + d_col*DBLOCK_SIZE + x].x;
			}
		}

		// init first cell of col
		if (d_col == 0) {
			tempLeftCell.y = 0;
			tempLeftCell.z = 0;
		} else {
			tempLeftCell.y = dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE].y;
			tempLeftCell.z = dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE].x;
		}

		// fill dynamic block
		for (x = 0; x < inner_row; x++) {
			for (y = 0; y < inner_col; y++) {
				left = tempLeftCell;
				up = scoreMatrixRow[y];
				diag = tempDiagCell;

				GET_SUB_SCORE(subScore, unpacked_query_batch[query_batch_offsets[tid] + DBLOCK_SIZE*d_col + y], unpacked_target_batch[target_batch_offsets[tid] + DBLOCK_SIZE*d_row + x]);
				tmpScore = diag.z + subScore;
				entry.x = max(up.x - _cudaGapExtend, up.z - _cudaGapOE);
				entry.y = max(left.y - _cudaGapExtend, left.z - _cudaGapOE);

				if (max(0, tmpScore) < max(entry.x, entry.y)) {
					if (entry.x < entry.y) {
						directionMatrix[DBLOCK_SIZE*x + y] = 3; // from left cell
						entry.z = entry.y;
					} else {
						directionMatrix[DBLOCK_SIZE*x + y] = 2; // from upper cell
						entry.z = entry.x;
					}
				} else {
					directionMatrix[DBLOCK_SIZE*x + y] = 0;
					entry.z = max(0, tmpScore); // from diagonal cell
				}

				tempLeftCell = entry;
				tempDiagCell = scoreMatrixRow[y];
				scoreMatrixRow[y] = entry;
			}
			
			if (d_col == 0) {
				tempLeftCell.y = 0;
				tempLeftCell.z = 0;
				tempDiagCell.y = 0;
				tempDiagCell.z = 0;
			} else {
				tempLeftCell.y = dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE + x + 1].y;
				tempLeftCell.z = dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE + x + 1].x;

				if (d_row == 0) {
					tempDiagCell.y = 0;
					tempDiagCell.z = 0;
				} else {
					tempDiagCell.y = dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE + x].y;
					tempDiagCell.z = dblock_col[dblock_offset + dp_mtx_len*d_col + d_row*DBLOCK_SIZE + x].x;
				}
			}
		}

		// traceback in a dynamic block
		while (inner_row >= 0 && inner_col >= 0) {
			direction = directionMatrix[DBLOCK_SIZE*inner_row + inner_col];

			switch(direction) {
				case 0: // matched
				case 1: // mismatched
					result_query[maximum_sequence_length*tid + query_matched_idx++] = unpacked_query_batch[query_batch_offsets[tid] + j];
					result_target[maximum_sequence_length*tid + target_matched_idx++] = unpacked_target_batch[target_batch_offsets[tid] + i];
					i--;
					j--;
					inner_row--;
					inner_col--;
				break;
				case 2: // from upper cell
					result_query[maximum_sequence_length*tid + query_matched_idx++] = unpacked_query_batch[query_batch_offsets[tid] + j];
					result_target[maximum_sequence_length*tid + target_matched_idx++] = '-';
					j--;
					inner_col--;
				break;
				case 3: // from left cell
					result_query[maximum_sequence_length*tid + query_matched_idx++] = '-';
					result_target[maximum_sequence_length*tid + target_matched_idx++] = unpacked_target_batch[target_batch_offsets[tid] + i];
					i--;
					inner_row--;
				break;
			}
		}

		if (inner_row < 0) {
			if (inner_col < 0) { // dblock diag
				d_row--;
				d_col--;
				inner_row = DBLOCK_SIZE-1;
				inner_col = DBLOCK_SIZE-1;
			} else { // dblock upper
				d_row--;
				inner_row = DBLOCK_SIZE-1;
			}
		} else { // dlbock left
			d_col--;
			inner_col = DBLOCK_SIZE-1;
		}
	}
}
#endif