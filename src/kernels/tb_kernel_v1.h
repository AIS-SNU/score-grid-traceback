#ifndef __TB_KERNEL__
#define __TB_KERNEL__

#define N_VALUE 'N'

#define GET_SUB_SCORE(subScore, query, target) \
	subScore = (query == target) ? _cudaMatchScore : -_cudaMismatchScore;\
	subScore = ((query == N_VALUE) || (target == N_VALUE)) ? 0 : subScore;\

__global__ void traceback_kernel(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *packed_tb_matrices, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length) {

	int i, j;
	int total_score __attribute__((unused));
	int curr_score __attribute__((unused));
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int tb_matrix_size = maximum_sequence_length*maximum_sequence_length/8;

	i = device_res->target_batch_end[tid];
	j = device_res->query_batch_end[tid];
	total_score = device_res->aln_score[tid];
	curr_score = 0;

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

__global__ void traceback_kernel_dynamic(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *packed_tb_matrices, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, short2 *dblock_row, short2 *dblock_col, uint8_t *local_direction, uint8_t *dblock_direction_global) {
// __global__ void traceback_kernel_dynamic(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *packed_tb_matrices, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, short2 *dblock_row, short2 *dblock_col, uint8_t *local_direction) {

	int i, j;
	int d_row, d_col;
	int inner_row, inner_col;
	int total_score __attribute__((unused));
	int curr_score __attribute__((unused));
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int tb_matrix_size = maximum_sequence_length*maximum_sequence_length/8;

	i = device_res->target_batch_end[tid];
	j = device_res->query_batch_end[tid];
	d_row = i/DBLOCK_SIZE;
	d_col = j/DBLOCK_SIZE;
	inner_row = i%DBLOCK_SIZE+1;
	inner_col = j%DBLOCK_SIZE+1;
	total_score = device_res->aln_score[tid];
	curr_score = 0;
	
	// printf("blockDim.x : %d\n", blockDim.x);
	// extern __shared__ uint8_t *sharedMemory;
	// int shm_offset = (tid%blockDim.x)*((sizeof(short3) + sizeof(uint8_t))/sizeof(uint8_t))*(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1);
	// printf("TEST1!!!!!!\n");
	// uint8_t *dblock_direction = &(sharedMemory[shm_offset]);
	// printf("TEST2!!!!!!\n");
	// short3 *dblock = (short3*)(&sharedMemory[shm_offset + (DBLOCK_SIZE+1)*(DBLOCK_SIZE+1)]);
	// printf("dblock_direction: %p, dblock: %p\n", dblock_direction, dblock);

	// extern __shared__ uint8_t sharedMemory[];

	// int shm_offset = (threadIdx.x)*(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1);
	// short3 *dblock = &sharedMemory[shm_offset];
	// uint8_t *dblock_direction = sharedMemory+(threadIdx.x)*(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1);
	
	// uint8_t *dblock_direction = &((uint8_t*)sharedMemory)[shm_offset];

	#define DBLOCK_SIZE 16
	short3 dblock[(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1)];
	uint8_t dblock_direction[(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1)];

	// __shared__ short3 dblock[17*17*32];
	// extern __shared__ uint8_t sharedMemory[];
	// __shared__ uint8_t sharedMemory[(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1)*128];
	// uint8_t *dblock_direction = sharedMemory+8*(threadIdx.x)*(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1);
	// short3 *dblock = (short3*)(dblock_direction+2*(DBLOCK_SIZE+1)*(DBLOCK_SIZE+1));


	int query_matched_idx = 0;
	int target_matched_idx = 0;

	int x, y;
	int direction;
	short subScore, tmpScore;
	short3 left, up, diag;

	short3 entry;
	int row_per_mtx = maximum_sequence_length/DBLOCK_SIZE;

	// if (tid == 0) {
	// 	printf("start query: %d, target: %d\n", j, i);
	// 	printf("query_batch_lens[0]: %d\n", query_batch_lens[0]);
	// 	printf("target_batch_lens[0]: %d\n", target_batch_lens[0]);
	// 	printf("init inner_row: %d, inner_col: %d\n", inner_row, inner_col);

	// 	printf("dblock_row:\n\t");
	// 	for(int i = 0; i < 16*10; i++) {
	// 		printf("%d ", dblock_row[maximum_sequence_length*row_per_mtx*3000 + maximum_sequence_length*1 + i].x);
	// 	}
	// 	printf("\n");
	// 	printf("dblock_col:\n\t");
	// 	for(int i = 0; i < 16*10; i++) {
	// 		printf("%d ", dblock_col[maximum_sequence_length*row_per_mtx*3000 + maximum_sequence_length*1 + i].x);
	// 	}
	// 	printf("\n");
	// }

	while (d_row >= 0 && d_col >= 0) {
		// init dynamic block
		dblock[0].x = d_col == 0 ? 0 : dblock_row[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_row + d_col*DBLOCK_SIZE - 1].y;
		dblock[0].y = d_row == 0 ? 0 : dblock_col[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_col + d_row*DBLOCK_SIZE - 1].y;
		dblock[0].z = d_row == 0 ? (d_col == 0 ? 0 : dblock_row[maximum_sequence_length*row_per_mtx*tid + d_col*DBLOCK_SIZE - 1].x) : (d_col == 0 ? dblock_col[maximum_sequence_length*row_per_mtx*tid + d_row*DBLOCK_SIZE - 1].x : dblock_row[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_row + d_col*DBLOCK_SIZE - 1].x);

		for (x = 1; x < DBLOCK_SIZE+1; x++) {
			if (d_row == 0) {
				dblock[x].x = 0;
				dblock[x].z = 0;
			} else {
				dblock[x].x = dblock_row[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_row + d_col*DBLOCK_SIZE + x - 1].y;
				dblock[x].z = dblock_row[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_row + d_col*DBLOCK_SIZE + x - 1].x;
			}

			if (d_col == 0) {
				dblock[x*(DBLOCK_SIZE+1)].y = 0;
				dblock[x*(DBLOCK_SIZE+1)].z = 0;
			} else {
				dblock[x*(DBLOCK_SIZE+1)].y = dblock_col[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_col + d_row*DBLOCK_SIZE + x - 1].y;
				dblock[x*(DBLOCK_SIZE+1)].z = dblock_col[maximum_sequence_length*row_per_mtx*tid + maximum_sequence_length*d_col + d_row*DBLOCK_SIZE + x - 1].x;
			}
		}

		// fill dynamic block
		for (x = 1; x <= inner_row; x++) { // target (row)
			for (y = 1; y <= inner_col; y++) { // query (col)
				left = dblock[(DBLOCK_SIZE+1)*x + y-1];
				up = dblock[(DBLOCK_SIZE+1)*(x-1) + y];
				diag = dblock[(DBLOCK_SIZE+1)*(x-1) + y-1];

				GET_SUB_SCORE(subScore, unpacked_query_batch[query_batch_offsets[tid] + DBLOCK_SIZE*d_col + y - 1], unpacked_target_batch[target_batch_offsets[tid] + DBLOCK_SIZE*d_row + x - 1]);
				tmpScore = diag.z + subScore;
				entry.x = max(up.x - _cudaGapExtend, up.z - _cudaGapOE);
				entry.y = max(left.y - _cudaGapExtend, left.z - _cudaGapOE);

				if (max(0, tmpScore) < max(entry.x, entry.y)) {
					if (entry.x < entry.y) {
						dblock_direction[DBLOCK_SIZE*x + y] = 3; // from left cell
						entry.z = entry.y;
					} else {
						dblock_direction[DBLOCK_SIZE*x + y] = 2; // from upper cell
						entry.z = entry.x;
					}
				} else {
					dblock_direction[DBLOCK_SIZE*x + y] = 0;
					entry.z = max(0, tmpScore); // from diagonal cell

				}
				
				dblock[(DBLOCK_SIZE+1)*x + y] = entry;
			}
		}


		// if (tid == 0 && d_row == 9 && d_col == 9) {
		// 	for (x = 0; x <= inner_row; x++) {
		// 		for (y = 0; y <= inner_col; y++) {
		// 			printf("[%d][%d]: %d, %d, %d, dir: %d\n", x, y, dblock[(DBLOCK_SIZE+1)*x + y].x, dblock[(DBLOCK_SIZE+1)*x + y].y, dblock[(DBLOCK_SIZE+1)*x + y].z, dblock_direction[DBLOCK_SIZE*x + y]);
		// 		}
		// 	}
		// }
		

		// traceback in dynamic block
		while (inner_row > 0 && inner_col > 0) {
			direction = dblock_direction[DBLOCK_SIZE*inner_row + inner_col];

			switch(direction) {
				case 0: // matched
				case 1: // mismatched
					result_query[maximum_sequence_length*tid + query_matched_idx++] = unpacked_query_batch[query_batch_offsets[tid] + j];
					result_target[maximum_sequence_length*tid + target_matched_idx++] = unpacked_target_batch[target_batch_offsets[tid] + i];
					i--;
					j--;
					inner_row--;
					inner_col--;
					// if (tid == 0 && d_row == 9 && d_col == 9) printf("GO DIAG!!\n");
				break;
				case 2: // from upper cell
					result_query[maximum_sequence_length*tid + query_matched_idx++] = unpacked_query_batch[query_batch_offsets[tid] + j];
					result_target[maximum_sequence_length*tid + target_matched_idx++] = '-';
					j--;
					inner_col--;
					// if (tid == 0 && d_row == 9 && d_col == 9) printf("GO LEFT!!\n");
				break;
				case 3: // from left cell
					result_query[maximum_sequence_length*tid + query_matched_idx++] = '-';
					result_target[maximum_sequence_length*tid + target_matched_idx++] = unpacked_target_batch[target_batch_offsets[tid] + i];
					i--;
					inner_row--;
					// if (tid == 0 && d_row == 9 && d_col == 9) printf("GO UPPER!!\n");
				break;
			}
		}

		if (inner_row == 0) {
			if (inner_col == 0) { // dblock diag
				d_row--;
				d_col--;
				inner_row = DBLOCK_SIZE;
				inner_col = DBLOCK_SIZE;
				// if (tid == 0) printf("[DBLOCK] GO DIAG!!\n");
			} else { // dblock upper
				d_row--;
				inner_row = DBLOCK_SIZE;
				// if (tid == 0) printf("[DBLOCK] GO UPPER!! inner_col: %d\n", inner_col);
			}
		} else { // dlbock left
			d_col--;
			inner_col = DBLOCK_SIZE;
			// if (tid == 0) printf("[DBLOCK] GO LEFT!! inner_row: %d\n", inner_row);
		}
		
	}
}
#endif