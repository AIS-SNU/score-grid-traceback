#ifndef __LOCAL_KERNEL_TEMPLATE__
#define __LOCAL_KERNEL_TEMPLATE__


// This old core provides the same result as the currently LOCAL core, but lacks some optimization. Left for historical / comparative purposes.
#define CORE_LOCAL_DEPRECATED_COMPUTE() \
		uint32_t gbase = (gpac >> l) & 15;/*get a base from target_batch sequence */ \
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);/* check equality of rbase and gbase */ \
		f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
		h[m] = p[m] + subScore; /*score if rbase is aligned to gbase*/ \
		h[m] = max(h[m], f[m]); \
		h[m] = max(h[m], 0); \
		e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
		h[m] = max(h[m], e); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1];

#define CORE_LOCAL_COMPUTE() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \

#define CORE_LOCAL_COMPUTE_START() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \

#define CORE_LOCAL_COMPUTE_TB(direction_reg) \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		uint32_t m_or_x = tmp_hm >= p[m] ? 0 : 1;\
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		direction_reg |= h[m] == tmp_hm ? m_or_x << (28 - ((m - 1) << 2)) : (h[m] == f[m] ? (uint32_t)3 << (28 - ((m - 1) << 2)) : (uint32_t)2 << (28 - ((m - 1) << 2)));\
		direction_reg |= (tmp_hm - _cudaGapOE) > (f[m] - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (31 - ((m - 1) << 2));\
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		direction_reg |= (tmp_hm - _cudaGapOE) > (e - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (30 - ((m - 1) << 2));\
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \




/* typename meaning : 
    - T is the algorithm type (LOCAL, MICROLOCAL)
    - S is WITH_ or WIHTOUT_START
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/
template <typename T, typename S, typename B>
__global__ void gasal_local_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_inter_row)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	//if (tid >= n_tasks) return;

	int32_t i, j, k, m, l, last_iter_len, job_per_thread;
	int32_t e;

    int32_t maxHH = 0; //initialize the maximum score to zero
	int32_t maxXY_y = 0; 

    int32_t prev_maxHH = 0;
    int32_t maxXY_x = 0;

    int32_t maxHH_second __attribute__((unused)); // __attribute__((unused)) to avoid raising errors at compilation. most template-kernels don't use these.
    int32_t prev_maxHH_second __attribute__((unused)); 
    int32_t maxXY_x_second __attribute__((unused));
    int32_t maxXY_y_second __attribute__((unused));
    maxHH_second = 0;
    prev_maxHH_second = 0;
    maxXY_x_second = 0;
    maxXY_y_second = 0;


	int32_t subScore;

	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	
	//-----arrays for saving intermediate values------
	//short2 global[MAX_QUERY_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	//--------------------------------------------


	//////////////////////////////////////////////
	int tx = threadIdx.x;
	int warp_len = 8;
	int warp_id = tx % warp_len; // id of warp in 
	int warp_block_id = tx / warp_len;
	int warp_num = tid / warp_len;
	int warp_per_kernel = (gridDim.x * blockDim.x) / warp_len; // number of warps. assume number of threads % warp_len == 0
	int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel ;
	//int warp_per_block = blockDim.x / warp_len; // number of warps in a block 
	
	// shared memory for intermediate values
	extern __shared__ short2 inter_row[];	//TODO: could use global mem instead
	int32_t* shared_maxHH = (int32_t*)(inter_row+((blockDim.x/8)*512));
	int job_per_query = max_query_len % warp_len ? (max_query_len / warp_len + 1) : max_query_len / warp_len;

	// start and end idx of sequences for each warp
	int job_start_idx = warp_num*job_per_warp;
	int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks;

	uint32_t packed_target_batch_idx, packed_query_batch_idx, read_len, ref_len, query_batch_regs, target_batch_regs;

	const int packed_len = 8;
	const int shared_len = 32;
	const int packed_warp_len = shared_len*2*packed_len;


	for (i = job_start_idx; i < job_end_idx; i++) {

		maxHH = 0; //initialize the maximum score to zero
		maxXY_y = 0; 

    	prev_maxHH = 0;
    	maxXY_x = 0;
		
		// get target and query seq
		packed_target_batch_idx = target_batch_offsets[i] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[i] >> 3;//starting index of the query_batch sequence
		read_len = query_batch_lens[i];
		ref_len = target_batch_lens[i];
		query_batch_regs = (read_len >> 3) + (read_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		target_batch_regs = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// fill with initial value
		for (j = 0; j < job_per_query; j++) {
			if ((j*warp_len + warp_id) < max_query_len) {
				global_inter_row[warp_num*max_query_len + j*warp_len + warp_id] = initHD;	
				//inter_row[warp_block_id*512 + j*warp_len + warp_id] = global_inter_row[warp_num*max_query_len + j*warp_len + warp_id];	
			}
		}
		
		int iter_num = target_batch_regs % warp_len ? (target_batch_regs / warp_len + 1):target_batch_regs / warp_len;

		//int check = 50;
		int up;
		for (j = 0; j < iter_num; j++) {
			// for each iteration on the target sequence
			for (m = 0; m < 9; m++) {
				h[m] = 0;
				f[m] = 0;
				p[m] = 0;
			}
			// get length of part of target seq
			int ref_iter_len = min(warp_len, target_batch_regs-j*warp_len);
			int longest_len = min(ref_iter_len, query_batch_regs);	//length of the longest anti-diag
			int longest_num = max(ref_iter_len, query_batch_regs) - longest_len + 1;	// number of the longest anti-diag
			int read_iter = 0;	//the element to read and save 
			up = 0;
			// remove idling threads
			register uint32_t gpac;
			if (warp_id < ref_iter_len) {
				// read packed element from ref that will be used in each thread only
				gpac = packed_target_batch[packed_target_batch_idx + j*warp_len + warp_id];
				gidx = (j*warp_len + warp_id) << 3;

			}
				

				// increasing phase
			int anti_len = 1;
			
			while(anti_len < longest_len) {
				if ((up % shared_len) == 0) {
					// read initial value from global memory
					
					for (ridx = 0; ridx < 8*(shared_len/warp_len); ridx++) {
						if ((ridx*warp_len + warp_id) < read_len) {
							inter_row[warp_block_id*packed_warp_len + ridx*warp_len + warp_id] = global_inter_row[warp_num*max_query_len + ridx*warp_len + warp_id];
						}
						
					}
					
				}
				if (warp_id < ref_iter_len && warp_id < anti_len) {					
					// read packed element from query
					register uint32_t rpac =packed_query_batch[packed_query_batch_idx + read_iter];
					ridx = 0;
					// 8*8 tile computation

					for (k = 28; k >= 0 && (ridx+read_iter*8) < read_len; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
						//-----load intermediate values--------------
						HD = inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx];
						//HD = global_inter_row[warp_num*max_query_len + (read_iter%(2*ref_iter_len))*packed_len + ridx];
						h[0] = HD.x;
						e = HD.y;

						#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {
							CORE_LOCAL_COMPUTE();
							if (SAMETYPE(B, Int2Type<TRUE>))
							{
								bool override_second = (maxHH_second < h[m]) && (maxHH > h[m]);
								maxXY_y_second = (override_second) ? gidx + (m-1) : maxXY_y_second;
								maxHH_second = (override_second) ? h[m] : maxHH_second;
							}
						}

						//----------save intermediate values------------
						HD.x = h[m-1];
						HD.y = e;
						inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx] = HD;
						//global_inter_row[warp_num*max_query_len + (read_iter%(2*ref_iter_len))*packed_len + ridx] = HD;
						//---------------------------------------------


						maxXY_x = (prev_maxHH < maxHH) ? ridx+read_iter*8 : maxXY_x;//end position on query_batch sequence corresponding to current maximum score

						if (SAMETYPE(B, Int2Type<TRUE>))
						{
							maxXY_x_second = (prev_maxHH_second < maxHH) ? ridx : maxXY_x_second;
							prev_maxHH_second = max(maxHH_second, prev_maxHH_second);
						}
						prev_maxHH = max(maxHH, prev_maxHH);
						ridx++;
						//-------------------------------------------------------

					}

					read_iter++;
					
				}
				up++;
				anti_len++;
			}

			// longest anit-diag phase
			int x;
			for (x = 0; x < longest_num; x++) {
				// all threads do 8*8 tile computation

				// read packed element from query
				register uint32_t rpac =packed_query_batch[packed_query_batch_idx + read_iter];
				
				if (up % shared_len == 0) {
					last_iter_len = min(up+shared_len, query_batch_regs);
					job_per_thread = (last_iter_len - up) % warp_len ? ((last_iter_len - up)/warp_len+1): (last_iter_len - up)/warp_len;
					for (ridx = 0; ridx < 8*job_per_thread; ridx++) {
						if ((up*8 + ridx*warp_len + warp_id) < max_query_len) {
							inter_row[warp_block_id*packed_warp_len + (up%(2*shared_len))*packed_len + ridx*warp_len + warp_id] = global_inter_row[warp_num*max_query_len + (up*packed_len) + ridx*warp_len + warp_id];
							
						}
					}
				}
				
				
				if (warp_id < ref_iter_len) {
					ridx = 0;
					// 8*8 tile computation

					for (k = 28; k >= 0 && (ridx+read_iter*8) < read_len; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
						//-----load intermediate values--------------
						HD = inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx];
						h[0] = HD.x;
						e = HD.y;

						#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {
							CORE_LOCAL_COMPUTE();
							if (SAMETYPE(B, Int2Type<TRUE>))
							{
								bool override_second = (maxHH_second < h[m]) && (maxHH > h[m]);
								maxXY_y_second = (override_second) ? gidx + (m-1) : maxXY_y_second;
								maxHH_second = (override_second) ? h[m] : maxHH_second;
							}
						}

						//----------save intermediate values------------
						HD.x = h[m-1];
						HD.y = e;
						inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx] = HD;
						//---------------------------------------------


						maxXY_x = (prev_maxHH < maxHH) ? ridx+read_iter*8 : maxXY_x;//end position on query_batch sequence corresponding to current maximum score

						if (SAMETYPE(B, Int2Type<TRUE>))
						{
							maxXY_x_second = (prev_maxHH_second < maxHH) ? ridx : maxXY_x_second;
							prev_maxHH_second = max(maxHH_second, prev_maxHH_second);
						}
						prev_maxHH = max(maxHH, prev_maxHH);
						ridx++;
						//-------------------------------------------------------

					}
					
				}
				
				read_iter++;
				up++;
				
				if ((x% shared_len) == (shared_len-1)) {
					for (ridx = 0; ridx < 8*(shared_len/warp_len); ridx++) {
						global_inter_row[warp_num*max_query_len + (x-shared_len+1)*packed_len + ridx*warp_len + warp_id] = inter_row[warp_block_id*packed_warp_len + ((x-shared_len+1)%(2*shared_len))*packed_len + ridx*warp_len + warp_id];
											
					}
					
				}
				

			}

			// decreasing 
			anti_len = 1;
			
			while (anti_len < longest_len) {
				if (warp_id < ref_iter_len && warp_id >= anti_len) {
					// 8*8 tile computation

					// read packed element from query
					register uint32_t rpac =packed_query_batch[packed_query_batch_idx + read_iter];

					ridx = 0;
					// 8*8 tile computation

					for (k = 28; k >= 0 && (ridx+read_iter*8) < read_len; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
						//-----load intermediate values--------------
						HD = inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx];
						h[0] = HD.x;
						e = HD.y;

						#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {
							CORE_LOCAL_COMPUTE();
							if (SAMETYPE(B, Int2Type<TRUE>))
							{
								bool override_second = (maxHH_second < h[m]) && (maxHH > h[m]);
								maxXY_y_second = (override_second) ? gidx + (m-1) : maxXY_y_second;
								maxHH_second = (override_second) ? h[m] : maxHH_second;
							}
						}

						//----------save intermediate values------------
						HD.x = h[m-1];
						HD.y = e;
						inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx] = HD;
						//---------------------------------------------


						maxXY_x = (prev_maxHH < maxHH) ? ridx+read_iter*8 : maxXY_x;//end position on query_batch sequence corresponding to current maximum score

						if (SAMETYPE(B, Int2Type<TRUE>))
						{
							maxXY_x_second = (prev_maxHH_second < maxHH) ? ridx : maxXY_x_second;
							prev_maxHH_second = max(maxHH_second, prev_maxHH_second);
						}
						prev_maxHH = max(maxHH, prev_maxHH);
						ridx++;
						//-------------------------------------------------------

					}
					

					read_iter++;
					
				}
				
				if ((x% shared_len) == (shared_len-1) ) {
					for (ridx = 0; ridx < 8*(shared_len/warp_len); ridx++) {
						if ((x-shared_len+1)*8 + ridx*warp_len + warp_id < read_len) {
							global_inter_row[warp_num*max_query_len + (x-shared_len+1)*packed_len + ridx*warp_len + warp_id] = inter_row[warp_block_id*packed_warp_len + ((x-shared_len+1)%(2*shared_len))*packed_len + ridx*warp_len + warp_id];
							
						}						
					}
					
				} else if (anti_len == (longest_len - 1)) {
					for (ridx = 0; ridx < 8*(shared_len/warp_len); ridx++) {
						if (((query_batch_regs-(query_batch_regs%shared_len))*8 + ridx*warp_len + warp_id) < max_query_len) {
							global_inter_row[warp_num*max_query_len + ((query_batch_regs-(query_batch_regs%shared_len))*packed_len) + ridx*warp_len + warp_id] = inter_row[warp_block_id*packed_warp_len + ((query_batch_regs-(query_batch_regs%shared_len))%(2*shared_len))*packed_len + ridx*warp_len + warp_id];
						}
					}
				
				}
				
				x++;
				anti_len++;
			}
		

		}

		__syncthreads();
		
		// add max values to shared memory 
	
		//reduction on max value
		shared_maxHH[tx] = maxHH;
		shared_maxHH[blockDim.x + tx] = maxXY_x;
		shared_maxHH[blockDim.x*2 + tx] = maxXY_y;


		for (int y=2; y <= warp_len; y*=2) {
			if (warp_id%y == 0 && warp_id < (warp_len - y/2)  ) {
				if (shared_maxHH[tx] < shared_maxHH[tx+y/2]  || (shared_maxHH[tx]==shared_maxHH[tx+y/2]&& shared_maxHH[blockDim.x*2+tx] > shared_maxHH[blockDim.x*2+tx+y/2])) {
					shared_maxHH[tx] = shared_maxHH[tx+y/2];
					shared_maxHH[blockDim.x + tx] = shared_maxHH[blockDim.x + tx+y/2];
					shared_maxHH[blockDim.x*2+tx] = shared_maxHH[blockDim.x*2+tx+y/2];
				}
			}
		}
		
		if (warp_id==0) {
			device_res->aln_score[i] = shared_maxHH[tx];//copy the max score to the output array in the GPU mem
			device_res->query_batch_end[i] = shared_maxHH[blockDim.x+tx];//copy the end position on query_batch sequence to the output array in the GPU mem
			device_res->target_batch_end[i] = shared_maxHH[blockDim.x*2+tx];//copy the end position on target_batch sequence to the output array in the GPU mem
			int i2 = i;
			

			if (SAMETYPE(B, Int2Type<TRUE>))
			{	//TODO: take care of second scores
				device_res_second->aln_score[i] = maxHH_second;
				device_res_second->query_batch_end[i] = maxXY_x_second;
				device_res_second->target_batch_end[i] = maxXY_y_second;
			}
		

			//TODO: 
			/*------------------Now to find the start position-----------------------*/
			if (SAMETYPE(S, Int2Type<WITH_START>))
			{

				int32_t rend_pos = device_res->query_batch_end[i];//end position on query_batch sequence
				int32_t gend_pos = device_res->target_batch_end[i];//end position on target_batch sequence
				int32_t fwd_score = device_res->aln_score[i];// the computed score

				//the index of 32-bit word containing the end position on query_batch sequence
				int32_t rend_reg = ((rend_pos >> 3) + 1) < query_batch_regs ? ((rend_pos >> 3) + 1) : query_batch_regs;
				//the index of 32-bit word containing to end position on target_batch sequence
				int32_t gend_reg = ((gend_pos >> 3) + 1) < target_batch_regs ? ((gend_pos >> 3) + 1) : target_batch_regs;
				


				packed_query_batch_idx += (rend_reg - 1);
				packed_target_batch_idx += (gend_reg - 1);


				maxHH = 0;
				prev_maxHH = 0;
				maxXY_x = 0;
				maxXY_y = 0;

				for (i = 0; i < max_query_len; i++) {
					global_inter_row[warp_num*max_query_len + i] = initHD;
				}
				//------starting from the gend_reg and rend_reg, align the sequences in the reverse direction and exit if the max score >= fwd_score------
				gidx = ((gend_reg << 3) + 8) - 1;
				
				for (i = 0; i < gend_reg && maxHH < fwd_score; i++) {
					for (m = 0; m < 9; m++) {
						h[m] = 0;
						f[m] = 0;
						p[m] = 0;
					}
					register uint32_t gpac =packed_target_batch[packed_target_batch_idx - i];//load 8 packed bases from target_batch sequence
					gidx = gidx - 8;
					ridx = (rend_reg << 3) - 1;
					int32_t global_idx = 0;
					
					for (j = 0; j < rend_reg && maxHH < fwd_score; j+=1) {
						register uint32_t rpac =packed_query_batch[packed_query_batch_idx - j];//load 8 packed bases from query_batch sequence
						//--------------compute a tile of 8x8 cells-------------------
						for (k = 0; k <= 28 && maxHH < fwd_score; k += 4) {
							uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
							//----------load intermediate values--------------
							HD = global_inter_row[warp_num*max_query_len + global_idx];
							h[0] = HD.x;
							e = HD.y;


							#pragma unroll 8
							for (l = 0, m = 1; l <= 28; l += 4, m++) {
									CORE_LOCAL_COMPUTE_START();
							}
							
							//------------save intermediate values----------------
							HD.x = h[m-1];
							HD.y = e;
							global_inter_row[warp_num*max_query_len + global_idx] = HD;
							//----------------------------------------------------
							maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//start position on query_batch sequence corresponding to current maximum score
							prev_maxHH = max(maxHH, prev_maxHH);
							ridx--;
							global_idx++;
						}
					}
					
				}
				
				device_res->query_batch_start[i2] = maxXY_x;//copy the start position on query_batch sequence to the output array in the GPU mem
				device_res->target_batch_start[i2] = maxXY_y;//copy the start position on target_batch sequence to the output array in the GPU mem

				i = i2;

			}
		}


	}
	



	///////////////////////////////////////////////

	



	return;


}
#endif
