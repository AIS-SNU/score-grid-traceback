#ifndef __TB_KERNEL__
#define __TB_KERNEL__

template <typename T>
__global__ void traceback_kernel(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *packed_tb_matrices, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length) {

	int i, j;
	int total_score __attribute__((unused));
	int curr_score __attribute__((unused));
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int tb_matrix_size = maximum_sequence_length*maximum_sequence_length/8;

	if (SAMETYPE(T, Int2Type<LOCAL>)) {
		i = device_res->target_batch_end[tid];
		j = device_res->query_batch_end[tid];
		total_score = device_res->aln_score[tid];
		curr_score = 0;
	} else { // TODO: needed?
		return;
	}

	int query_matched_idx = 0;
	int target_matched_idx = 0;

	while (i >= 0 && j >= 0) {
		int direction = (packed_tb_matrices[tb_matrix_size*tid + maximum_sequence_length*(i>>3) + j] >> (28 - 4*(i & 7))) & 3;

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
#endif