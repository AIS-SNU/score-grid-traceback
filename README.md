# Score Grid Traceback
It is a grad-project code for GPU-Accelerated DNA Sequence Alignment.

It is based on GASAL2(https://github.com/nahmedraja/GASAL2) and SALoBa(https://github.com/AIS-SNU/saloba).

For implementation details of Score Grid Traceback, see [score_grid_tb.pdf](paper/score_grid_tb.pdf)

## Features
- Code for storing data needed to execute traceback. (in local_kernel_template.h)
- Traceback kernel. (tb_kernel.h)
- Toggle between baseline and new kernel with a macro setting. (gasal_aligh.h DYNAMIC_TB)

## Requirements
A Linux platform with CUDA toolkit 8 or higher is required, along with usual build environment for C and C++ code. GASAL2 has been tested over NVIDIA GPUs with compute capabilities of 2.0, 3.5 and 5.0. Although lower versions of the CUDA framework might work, they have not been tested.

## Compiling GASAL2
The library can be compiled with the following two commands:

```bash
$ ./configure.sh <path to cuda installation directory>
$ make GPU_SM_ARCH=<GPU SM architecture> MAX_QUERY_LEN=<maximum query length> N_CODE=<code for "N", e.g. 0x4E if the bases are represented by ASCII characters> [N_PENALTY=<penalty for aligning "N" against any other base>]
```

`N_PENALTY` is optional and if it is not specified then GASAL2 considers "N" as an ordinary base having the same match/mismatch scores as for A, C, G or T. As a result of these commands, *include* and *lib* directories will be created containing various `.h` files and `libgasal.a`, respectively. The user needs to include `gasal_header.h` in the code and link it with `libgasal.a` during compilation. Also, the CUDA runtime library has to be linked by adding `-lcudart` flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*.

## Running Test Program
You can compile and run test_prog and get the result.

### Build the program:
```bash
cd test_prog
make
```

### ./test_prog.out -h shows:
```bash
Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-t] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>
Options: -a INT    match score [1]
         -b INT    mismatch penalty [4]
         -q INT    gap open penalty [6]
         -r INT    gap extension penalty [1]
         -s        find the start position
         -t        compute traceback. With this option enabled, "-s" has no effect as start position will always be computed with traceback
         -p        print the alignment results
         -n INT    Number of threads [1]
         -y AL_TYPE       Alignment type . Must be "local", "semi_global", "global", "ksw"
         -x HEAD TAIL     specifies, for semi-global alignment, wha should be skipped for heads and tails of the sequences. (NONE, QUERY, TARGET, BOTH)
         -k INT    Band width in case "banded" is selected.
         --help, -h : displays this message.
         --second-best   displays second best score (WITHOUT_START only).
Single-pack multi-Parameters (e.g. -sp) is not supported.
```

### Example of running test_prog.out
```bash
./test_prog.out -a 3 -b 3 -q 4 -r 1 -n 1 -t -y {query_fasta_file_route} {reference_fasta_file_route}
```

The final results are copied into

```cpp
uint8_t *result_query_host, *result_target_host
```

as below:
```bash
[0]: ACCCCTACACC-GTCAAGTTCCGAGC
[1]: CACCATGAAG-TA-A--GCT-AA-----CGTAC
...
[STREAM_BATCH_SIZE-1]: ...
```


## check_result.out
You can see the result of CPU calculated traceback result. It's not for comparing process time but for result consistency.

// FIXME:
There may be some differences in results since the logic of selecting maximum score cell is slightly different from local_kernel_template.h when there are multiple maximum score cells.

Change the values below:

Base pairs having idx in [target_batch_idx, target_batch_idx + check_size) are calculated.
```cpp
int target_batch_idx;
bool show_inputs;
bool show_results;
int check_size;
```

### Running Example
```bash
cd test_prog
make check_result.out
./check_result.out {query_fasta_file_route} {reference_fasta_file_route}

query input: {query_fasta_file_name}, target input: {reference_fasta_file_name}

target_batch_idx 0 processed
Inputs[0]:
	query (length: 25):	CGAGCCTTGAACTGCCACATCCCCA
	target (length: 48):	CGAGCCTTGAAGCCTCCCCAACCCCAATACCCTGCCGCTTCACCCTAA
Results[0]:
	query:	ACCCCTACACC-GTCAAGTTCCGAGC
	target:	ACCCCAACCCCTCCGAAGTTCCGAGC

Inputs[1]:
	query (length: 40):	TCGAAATAAAGTAACATGCAATCGAATGAAGTACCACTGA
	target (length: 78):	TCGAAATAAAAACTCCAAACCCGGCGGTACTACGTAGACGGAAGACGGAAACATCCTCAAAACCGAGGAGTACGGATA
Results[1]:
	query:	CACCATGAAG-TA-A--GCT-AA-----CGTAC
	target:	CATCATGGCGGCCCAAACCTCAAAAATAAAGCT

Inputs[2]:
	query (length: 57):	AAGCGATCTAGCAAAAGAAAGGATTTCTGCGCATACCGATTCCCGTCCTGCACGATC
	target (length: 112):	AAGCGATCTAGCAAAGAAAGGATTTCTTGATAGTAGACGGCTTTCAACTACACGTAGCATCTGACCCAATATAAGGGAAGGGCCCCAAACACGTAAGCATAAAACCGCGGAC
Results[2]:
	query:	CTAGCACGT-CCTGCCCTTAGCCAT-ACGCGTCTTTAGGAAAGAAAACGATCTAGCGAA
	target:	CTA-C--GATGC-AC----ATCAA-CTTTCG------G--CAGA------TG-ATAGTT
```