

#include "../include/gasal_header.h"


#include <vector>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "Timer.h"

#define NB_STREAMS 1

//#define STREAM_BATCH_SIZE (262144)
// this gives each stream HALF of the sequences.
//#define STREAM_BATCH_SIZE ceil((double)target_seqs.size() / (double)(2))

#define STREAM_BATCH_SIZE 10000 //ceil((double)target_seqs.size() / (double)(2 * 2))

#define DEBUG

#define MAX(a,b) (a>b ? a : b)


//#define GPU_SELECT 0

int main(int argc, char **argv) {	
	Timer local_time;
	Timer malloc_time;
	Timer free_time;
	cudaDeviceSynchronize();
	Timer total_time;
	total_time.Start();
	Timer load_time;
	load_time.Start();

	//gasal_set_device(GPU_SELECT);

	Parameters *args;
	args = new Parameters(argc, argv);
	// args->isPacked = true;
	args->parse();
	args->print();

	cudaError_t err;

	int n_threads = args->n_threads;

	//--------------copy substitution scores to GPU--------------------
	gasal_subst_scores sub_scores;

	sub_scores.match = args->sa;
	sub_scores.mismatch = args->sb;
	sub_scores.gap_open = args->gapo;
	sub_scores.gap_extend = args->gape;

	gasal_copy_subst_scores(&sub_scores);

	//-------------------------------------------------------------------


	std::vector<std::string> query_seqs;
	std::vector<std::string> target_seqs;
	std::vector<std::string> query_headers;
	std::vector<std::string> target_headers;
	std::string query_batch_line, target_batch_line;

	int total_seqs = 0;
	uint32_t maximum_sequence_length = 0;
	uint32_t target_seqs_len = 0;
	uint32_t query_seqs_len = 0;
	std::cerr << "Loading files...." << std::endl;

	/*
		Reads FASTA files and fill the corresponding buffers.
		FASTA files contain sequences that are usually on separate lines.
		The file reader detects a '>' then concatenates all the following lines into one sequence, until the next '>' or EOF.
		See more about FASTA format : https://en.wikipedia.org/wiki/FASTA_format
	*/
	
	int seq_begin=0;

	std::vector<uint8_t> query_mod;
	std::vector<uint8_t> target_mod;
	std::vector<uint32_t> query_id;
	std::vector<uint32_t> target_id;

	char line_starts[5] = "></+";
	/* The information of reverse-complementing is simulated by changing the first character of the sequence.
	 * This is not explicitly FASTA-compliant, although regular FASTA files will simply be interpreted as Forward-Natural direction.
	 * From the header of every sequence:
	 * - '>' translates to 0b00 (0) = Forward, natural
	 * - '<' translates to 0b01 (1) = Reverse, natural
	 * - '/' translates to 0b10 (2) = Forward, complemented
	 * - '+' translates to 0b11 (3) = Reverse, complemented
	 * No protection is done, so any other number will only have its two first bytes counted as above.	 
	 */

	while (getline(args->query_batch_fasta, query_batch_line) && getline(args->target_batch_fasta, target_batch_line)) { 

		//load sequences from the files
		char *q = NULL;
		char *t = NULL;
		q = strchr(line_starts, (int) (query_batch_line[0]));
		t = strchr(line_starts, (int) (target_batch_line[0]));

		/*  
			t and q are pointers to the first occurence of the first read character in the line_starts array.
			so if I compare the address of these pointers with the address of the pointer to line_start, then...
			I can get which character was found, so which modifier is required. 
		*/

		if (q != NULL && t != NULL) {
			total_seqs++;

			query_mod.push_back((uint8_t) (q-line_starts));
			query_id.push_back(total_seqs);

			target_mod.push_back((uint8_t)(t-line_starts));
			target_id.push_back(total_seqs);

			query_headers.push_back(query_batch_line.substr(1));
			target_headers.push_back(target_batch_line.substr(1));

			if (seq_begin == 2) {
				// a sequence was already being read. Now it's done, so we should find its length.
				target_seqs_len += (target_seqs.back()).length();
				query_seqs_len += (query_seqs.back()).length();
				maximum_sequence_length = MAX((target_seqs.back()).length(), maximum_sequence_length);
				maximum_sequence_length = MAX((query_seqs.back()).length(), maximum_sequence_length);
			}
			seq_begin = 1;
			
		} else if (seq_begin == 1) {
			query_seqs.push_back(query_batch_line);
			target_seqs.push_back(target_batch_line);
			seq_begin=2;
		} else if (seq_begin == 2) {
			query_seqs.back() += query_batch_line;
			target_seqs.back() += target_batch_line;
		} else { // should never happen but always put an else, for safety...
			seq_begin = 0;
			std::cerr << "Batch1 and target_batch files should be fasta having same number of sequences" << std::endl;
			exit(EXIT_FAILURE);
		}
	}



	// Check maximum sequence length one more time, to check the last read sequence:
	target_seqs_len += (target_seqs.back()).length();
	query_seqs_len += (query_seqs.back()).length();
	maximum_sequence_length = MAX((target_seqs.back()).length(), maximum_sequence_length);
	maximum_sequence_length = MAX((query_seqs.back()).length(), maximum_sequence_length);
	int maximum_sequence_length_query = MAX((query_seqs.back()).length(), 0);

	#ifdef DEBUG
		std::cerr << "[TEST_PROG DEBUG]: ";
		std::cerr << "Size of read batches are: query=" << query_seqs_len << ", target=" << target_seqs_len << ". maximum_sequence_length=" << maximum_sequence_length << std::endl;
	#endif
	load_time.Stop();

	Timer distr_time;
	distr_time.Start();

	// transforming the _mod into a char* array (to be passed to GASAL, which deals with C types)
	uint8_t *target_seq_mod = (uint8_t*) malloc(total_seqs * sizeof(uint8_t) );
	uint8_t *query_seq_mod  = (uint8_t*) malloc(total_seqs * sizeof(uint8_t) );
	uint32_t *target_seq_id = (uint32_t*) malloc(total_seqs * sizeof(uint32_t) );
	uint32_t *query_seq_id  = (uint32_t*) malloc(total_seqs * sizeof(uint32_t) );

	for (int i = 0; i < total_seqs; i++)
	{
		query_seq_mod[i] = query_mod.at(i);
		query_seq_id[i] = query_id.at(i);
	}

#ifdef DEBUG
	std::cerr << "[TEST_PROG DEBUG]: query, mod@id=";
	for (int i = 0; i < total_seqs; i++)
	{
		if ((query_seq_mod[i]) > 0)
			std::cerr << +(query_seq_mod[i]) << "@" << query_seq_id[i] << "| ";
	}
	
	std::cerr << std::endl;
#endif

	for (int i = 0; i < total_seqs; i++)
	{
		target_seq_mod[i] = target_mod.at(i);
		target_seq_id[i] = target_id.at(i);
	}

	int *thread_seqs_idx = (int*)malloc(n_threads*sizeof(int));
	int *thread_n_seqs = (int*)malloc(n_threads*sizeof(int));
	int *thread_n_batchs = (int*)malloc(n_threads*sizeof(int));
	double *thread_misc_time = (double*)calloc(n_threads, sizeof(double));

	int thread_batch_size = (int)ceil((double)total_seqs/n_threads);
	int n_seqs_alloc = 0;
	for (int i = 0; i < n_threads; i++){//distribute the sequences among the threads equally
		thread_seqs_idx[i] = n_seqs_alloc;
		if (n_seqs_alloc + thread_batch_size < total_seqs) thread_n_seqs[i] = thread_batch_size;
		else thread_n_seqs[i] = total_seqs - n_seqs_alloc;
		thread_n_batchs[i] = (int)ceil((double)thread_n_seqs[i]/(STREAM_BATCH_SIZE));
		n_seqs_alloc += thread_n_seqs[i];
	}
	distr_time.Stop();

	std::cerr << "Processing..." << std::endl;

	Timer process_time;
	process_time.Start();
	omp_set_num_threads(n_threads);
	gasal_gpu_storage_v *gpu_storage_vecs =  (gasal_gpu_storage_v*)calloc(n_threads, sizeof(gasal_gpu_storage_v));
	for (int z = 0; z < n_threads; z++) {
		gpu_storage_vecs[z] = gasal_init_gpu_storage_v(NB_STREAMS);// creating NB_STREAMS streams per thread
		
		/* 
			About memory sizes:
			The required memory is the total size of the batch + its padding, divided by the number of streams. 
			The worst case would be that every sequence has to be padded with 7 'N', since they must have a length multiple of 8.
			Even though the memory can be dynamically expanded both for Host and Device, it is advised to start with a memory large enough so that these expansions rarely occur (for better performance.)
			Modifying the factor '1' in front of each size lets you see how GASAL2 expands the memory when needed.
		*/
		/*
		// For exemple, this is exactly the memory needed to allocate to fit all sequences is a single GPU BATCH.
		gasal_init_streams(&(gpu_storage_vecs[z]), 
							1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) , 
							1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) , 
							1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
							1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS))  , 
							ceil((double)target_seqs.size() / (double)(NB_STREAMS)), // maximum number of alignments is bigger on target than on query side.
							ceil((double)target_seqs.size() / (double)(NB_STREAMS)), 
							args);
		*/		
		//initializing the streams by allocating the required CPU and GPU memory
		// note: the calculations of the detailed sizes to allocate could be done on the library side (to hide it from the user's perspective)
		gasal_init_streams(&(gpu_storage_vecs[z]), (maximum_sequence_length_query + 7) , //TODO: remove maximum_sequence_length_query
						(maximum_sequence_length + 7) ,
						 STREAM_BATCH_SIZE, //device
						 args);
	}
	#ifdef DEBUG
		std::cerr << "[TEST_PROG DEBUG]: ";
		std::cerr << "size of host_unpack_query is " << (query_seqs_len +7*total_seqs) / (NB_STREAMS) << std::endl ;
	#endif

	#pragma omp parallel
	{
	int n_seqs = thread_n_seqs[omp_get_thread_num()];//number of sequences allocated to this thread
	int curr_idx = thread_seqs_idx[omp_get_thread_num()];//number of sequences allocated to this thread
	int seqs_done = 0;
	int n_batchs_done = 0;

	struct gpu_batch{ //a struct to hold data structures of a stream
			gasal_gpu_storage_t *gpu_storage; //the struct that holds the GASAL2 data structures
			int n_seqs_batch;//number of sequences in the batch (<= (target_seqs.size() / NB_STREAMS))
			int batch_start;//starting index of batch
	};

	#ifdef DEBUG
		std::cerr << "[TEST_PROG DEBUG]: ";
		std::cerr << "Number of gpu_batch in gpu_batch_arr : " << gpu_storage_vecs[omp_get_thread_num()].n << std::endl;
		std::cerr << "[TEST_PROG DEBUG]: ";
		std::cerr << "Number of gpu_storage_vecs in a gpu_batch : " << omp_get_thread_num()+1 << std::endl;
	#endif

	gpu_batch gpu_batch_arr[gpu_storage_vecs[omp_get_thread_num()].n];

	for(int z = 0; z < gpu_storage_vecs[omp_get_thread_num()].n; z++) {
		gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[omp_get_thread_num()].a[z]);

	}

	uint32_t BLOCKDIM = 64;
	uint32_t N_BLOCKS = (STREAM_BATCH_SIZE + BLOCKDIM - 1) / BLOCKDIM;
	
	short2* global_inter_row;
	malloc_time.Start();
	cudaMalloc((void**) &global_inter_row, sizeof(short2)*maximum_sequence_length*(BLOCKDIM/8)*N_BLOCKS);
	malloc_time.Pause();
	
	uint32_t* global_direction;
	uint8_t *result_query;
	uint8_t *result_target;
	short2 *dblock_row, *dblock_col;
	uint8_t *result_query_host;
	uint8_t *result_target_host;

	// calculate dblock_row, dblock_col size
	int gb_idx = 0;
	int64_t dblock_matrix_size = 0;
	int64_t global_direction_size = 0;

	for (int z = 0; z < thread_n_batchs[omp_get_thread_num()]; z++) {
		int sum = 0;
		int idx_in_batch = 0;

		for (idx_in_batch = 0; idx_in_batch < STREAM_BATCH_SIZE && gb_idx < n_seqs; idx_in_batch++, gb_idx++) {
			#ifdef DYNAMIC_TB
			sum += (MAX(query_seqs[gb_idx].size(), target_seqs[gb_idx].size()) * MAX(query_seqs[gb_idx].size(), target_seqs[gb_idx].size()) + DBLOCK_SIZE - 1) / DBLOCK_SIZE;
			#else
			sum += MAX(query_seqs[gb_idx].size(), target_seqs[gb_idx].size()) * ((MAX(query_seqs[gb_idx].size(), target_seqs[gb_idx].size())+7) / 8);
			#endif
		}
		#ifdef DYNAMIC_TB
		dblock_matrix_size = MAX(dblock_matrix_size, sum);
		#else
		global_direction_size = MAX(global_direction_size, sum);
		#endif
		
	}


	if (args->start_pos == WITH_TB) {
		malloc_time.Start();

		#ifdef DYNAMIC_TB
		CHECKCUDAERROR(cudaMalloc((void**) &dblock_row, sizeof(short2)*dblock_matrix_size));
		CHECKCUDAERROR(cudaMalloc((void**) &dblock_col, sizeof(short2)*dblock_matrix_size));
		#else
		// cudaMalloc((void**) &global_direction, sizeof(uint32_t)*maximum_sequence_length*maximum_sequence_length*STREAM_BATCH_SIZE/8);
		printf("global direction size: %ld\n", global_direction_size);
		CHECKCUDAERROR(cudaMalloc((void**) &global_direction, sizeof(uint32_t)*global_direction_size));
		#endif
		
		CHECKCUDAERROR(cudaMalloc((void**) &result_query, sizeof(uint8_t)*maximum_sequence_length*STREAM_BATCH_SIZE));
		CHECKCUDAERROR(cudaMalloc((void**) &result_target, sizeof(uint8_t)*maximum_sequence_length*STREAM_BATCH_SIZE));
		result_query_host = (uint8_t*)malloc(sizeof(uint8_t)*maximum_sequence_length*STREAM_BATCH_SIZE);
		result_target_host = (uint8_t*)malloc(sizeof(uint8_t)*maximum_sequence_length*STREAM_BATCH_SIZE);
		malloc_time.Pause();
	}

	if (n_seqs > 0) {
		while (n_batchs_done < thread_n_batchs[omp_get_thread_num()]) { // Loop on streams
			int gpu_batch_arr_idx = 0;
			//------------checking the availability of a "free" stream"-----------------
			while(gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n && (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->is_free != 1) {
				gpu_batch_arr_idx++;
			}

			if (seqs_done < n_seqs && gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n) {
				uint32_t query_batch_idx = 0;
				uint32_t target_batch_idx = 0;
				uint32_t dp_matrix_idx = 0;
				uint64_t global_direction_idx = 0;
				unsigned int j = 0;
				//-----------Create a batch of sequences to be aligned on the GPU. The batch contains (target_seqs.size() / NB_STREAMS) number of sequences-----------------------

				for (int i = curr_idx; seqs_done < n_seqs && j < (STREAM_BATCH_SIZE); i++, j++, seqs_done++)
				{

					gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns++ ;

					if(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns > gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns)
					{
						gasal_host_alns_resize(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns * 2, args);
					}

					(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j] = query_batch_idx;
					(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_offsets[j] = target_batch_idx;

					/*
						All the filling is moved on the library size, to take care of the memory size and expansions (when needed).
						The function gasal_host_batch_fill takes care of how to fill, how much to pad with 'N', and how to deal with memory. 
						It's the same function for query and target, and you only need to set the final flag to either ; this avoides code duplication.
						The way the host memory is filled changes the current _idx (it's increased by size, and by the padding). That's why it's returned by the function.
					*/

					query_batch_idx = gasal_host_batch_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, 
									query_batch_idx, 
									query_seqs[i].c_str(), 
									query_seqs[i].size(),
									QUERY);

					target_batch_idx = gasal_host_batch_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, 
									target_batch_idx, 
									target_seqs[i].c_str(), 
									target_seqs[i].size(),
									TARGET);

					
					(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_lens[j] = query_seqs[i].size();
					(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_lens[j] = target_seqs[i].size();

					(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->dp_matrix_offsets[j] = dp_matrix_idx;
					(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->global_direction_offsets[j] = global_direction_idx;

					dp_matrix_idx += (MAX(query_seqs[i].size(), target_seqs[i].size()) * MAX(query_seqs[i].size(), target_seqs[i].size()) + DBLOCK_SIZE - 1) / DBLOCK_SIZE;
					global_direction_idx += MAX(query_seqs[i].size(), target_seqs[i].size()) * ((MAX(query_seqs[i].size(), target_seqs[i].size())+7) / 8);

				}


				#ifdef DEBUG
					std::cerr << "[TEST_PROG DEBUG]: ";
					std::cerr << "Stream " << gpu_batch_arr_idx << ": j = " << j << ", seqs_done = " << seqs_done <<", query_batch_idx=" << query_batch_idx << ", target_batch_idx=" << target_batch_idx << ", dp_matrix_idx=" << dp_matrix_idx << std::endl;
				#endif
				

				// Here, we fill the operations arrays for the current batch to be processed by the stream
				gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod + seqs_done - j, j, QUERY);
				gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod + seqs_done - j, j, TARGET);


				gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = j;
				uint32_t query_batch_bytes = query_batch_idx;
				uint32_t target_batch_bytes = target_batch_idx;
				gpu_batch_arr[gpu_batch_arr_idx].batch_start = curr_idx;
				curr_idx += (STREAM_BATCH_SIZE);

				//----------------------------------------------------------------------------------------------------
				//-----------------calling the GASAL2 non-blocking alignment function---------------------------------
				
				local_time.Start();
				gasal_aln_async(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args, maximum_sequence_length, global_inter_row, global_direction, result_query, result_target, dblock_row, dblock_col);
				
				local_time.Stop();
				gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;
				//---------------------------------------------------------------------------------
			}

			//-------------------------------print alignment results----------------------------------------
		
			gpu_batch_arr_idx = 0;
			while (gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n) {//loop through all the streams and print the results
																					//of the finished streams.
				if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
					// copy results from device
					CHECKCUDAERROR(cudaMemcpy(result_query_host, result_query, sizeof(uint8_t)*maximum_sequence_length*STREAM_BATCH_SIZE, cudaMemcpyDeviceToHost));
					CHECKCUDAERROR(cudaMemcpy(result_target_host, result_target, sizeof(uint8_t)*maximum_sequence_length*STREAM_BATCH_SIZE, cudaMemcpyDeviceToHost));

					//----------------------------------------------------------------------

					n_batchs_done++;
				}
				gpu_batch_arr_idx++;
			}
		}
	}

	
	free_time.Start();
	CHECKCUDAERROR(cudaFree(global_inter_row));

	if (args->start_pos == WITH_TB) {
		
		#ifdef DYNAMIC_TB
		CHECKCUDAERROR(cudaFree(dblock_row));
		CHECKCUDAERROR(cudaFree(dblock_col));
		#else
		CHECKCUDAERROR(cudaFree(global_direction));
		#endif		

		CHECKCUDAERROR(cudaFree(result_query));
		CHECKCUDAERROR(cudaFree(result_target));
		free(result_query_host);
		free(result_target_host);
		
	}
	free_time.Pause();

	}
	for (int z = 0; z < n_threads; z++) {
		gasal_destroy_streams(&(gpu_storage_vecs[z]), args);
		gasal_destroy_gpu_storage_v(&(gpu_storage_vecs[z]));
	}
	free(gpu_storage_vecs);

	

	process_time.Stop();
	/*
	string algorithm = al_type;
	string start_type[2] = {"without_start", "with_start"};
	al_type += "_";
	al_type += start_type[start_pos==WITH_START];
	*/
	double av_misc_time = 0.0;
	for (int i = 0; i < n_threads; ++i){
		av_misc_time += (thread_misc_time[i]/n_threads);
	}
	std::cerr << std::endl << "Done" << std::endl;
	//fprintf(stderr, "Total execution time (in milliseconds): %.3f\n", total_time.GetTime());
	delete args; // closes the files
	//free(args); // closes the files
	total_time.Stop();
	fprintf(stderr, "load time (in milliseconds): %.3f\n", load_time.GetTime());
	fprintf(stderr, "distribution time (in milliseconds): %.3f\n", distr_time.GetTime());
	fprintf(stderr, "process time (with malloc) (in milliseconds): %.3f\n", process_time.GetTime());
	fprintf(stderr, "malloc time (in milliseconds): %.3f\n", malloc_time.GetTime());
	fprintf(stderr, "free time (in milliseconds): %.3f\n", free_time.GetTime());
	fprintf(stderr, "local kernel time (in milliseconds): %.3f\n", local_time.GetTime());
	fprintf(stderr, "total time (in milliseconds): %.3f\n", total_time.GetTime());

}
