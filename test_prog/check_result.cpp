#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#define MATCH_SCORE 3
#define MISMATCH_SCORE 3
#define GAP_OPEN_SCORE 5
#define GAP_EXTENSION_SCORE 1

#define GET_SUB_SCORE(subScore, query, target) \
	subScore = (query == target) ? MATCH_SCORE : -MISMATCH_SCORE;\

struct short3
{
    short x, y, z;
};

/**
 * There may be some differences in results since the logic of selecting maximum score cell
 * is slightly different from local_kernel_template.h when there are multiple maximum score cells.
 */
int main(int argc, char **argv) {
    std::ifstream query_batch_fasta;
    std::ifstream target_batch_fasta;
    std::string query_batch_fasta_filename = std::string( (const char*)  (*(argv + 1) ) );
    std::string target_batch_fasta_filename = std::string( (const char*)  (*(argv + 2) ) );
    std::string query_batch_line, target_batch_line;
    std::cout << "query input: " << query_batch_fasta_filename << ", target input: " << target_batch_fasta_filename << std::endl << std::endl;

    query_batch_fasta.open(query_batch_fasta_filename, std::ifstream::in);
    target_batch_fasta.open(target_batch_fasta_filename, std::ifstream::in);

    if (!query_batch_fasta || !target_batch_fasta) {
        std::cout << "ERROR!!" << std::endl;
        return 1;
    }

    //---------------- Change it to debug --------------------
    // Base pairs having idx in [target_batch_idx, target_batch_idx + check_size) are calculated
    int target_batch_idx = 0;
    bool show_inputs = true;
    bool show_results = true;
    int check_size = 10;
    //--------------------------------------------------------
    int target_line_num = 2*target_batch_idx;
    int count = 0;

    #define MAX_SEQ_LEN 10000
    short3 **scoreMatrix = new short3*[MAX_SEQ_LEN];
    uint8_t **directionMatrix = new uint8_t*[MAX_SEQ_LEN];

    for (int x = 0; x < MAX_SEQ_LEN; x++) {
        scoreMatrix[x] = new short3[MAX_SEQ_LEN];
        directionMatrix[x] = new uint8_t[MAX_SEQ_LEN];
    }

    while (count++ < target_line_num) {
        getline(query_batch_fasta, query_batch_line);
        getline(target_batch_fasta, target_batch_line);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int x = 0; x < check_size; x++) {
        getline(query_batch_fasta, query_batch_line);
        getline(target_batch_fasta, target_batch_line);

        if (!(getline(query_batch_fasta, query_batch_line) && getline(target_batch_fasta, target_batch_line))) {
            std::cout << "No more line, terminated." << std::endl;
            return 1;
        }

        if (target_batch_idx % 10 == 0) {
            std::cout << "target_batch_idx " << target_batch_idx << " processing" << std::endl;
        }

        if (show_inputs) {
            std::cout << "Inputs[" << target_batch_idx << "]:" << std::endl;
            std::cout << "\tquery (length: " << query_batch_line.length() << "):\t" << query_batch_line << std::endl;
            std::cout << "\ttarget (length: " << target_batch_line.length() << "):\t" << target_batch_line << std::endl;
        }

        int i, j;
        short3 left, up, diag;
        int subScore, tmpScore;
        short3 entry;
        int maxRow = 0, maxCol = 0, maxScore = 0;
        
        for (i = 0; i < target_batch_line.length()+1; i++) {
            for (j = 0; j < query_batch_line.length()+1; j++) {
                scoreMatrix[i][j].x = 0;
                scoreMatrix[i][j].y = 0;
                scoreMatrix[i][j].z = 0;
                directionMatrix[i][j] = 0;
            }
        }

        for (i = 1; i < target_batch_line.length()+1; i++) {
            for (j = 1; j < query_batch_line.length()+1; j++) {
                left = scoreMatrix[i][j-1];
                up = scoreMatrix[i-1][j];
                diag = scoreMatrix[i-1][j-1];

                GET_SUB_SCORE(subScore, query_batch_line[j-1], target_batch_line[i-1]);
                tmpScore = diag.z + subScore;
                entry.x = std::max(up.x - GAP_EXTENSION_SCORE, up.z - GAP_OPEN_SCORE);
                entry.y = std::max(left.y - GAP_EXTENSION_SCORE, left.z - GAP_OPEN_SCORE);

                if (std::max(0, tmpScore) < std::max(entry.x, entry.y)) {
                    if (entry.x < entry.y) {
                        directionMatrix[i][j] = 3; // from left cell
                        entry.z = entry.y;
                    } else {
                        directionMatrix[i][j] = 2; // from upper cell
                        entry.z = entry.x;
                    }
                } else {
                    directionMatrix[i][j] = 0;
                    entry.z = std::max(0, tmpScore); // from diagonal cell

                }
                scoreMatrix[i][j] = entry;

                if (entry.z > maxScore) {
                    maxRow = i;
                    maxCol = j;
                    maxScore = entry.z;
                }
            }
        }

        i = maxRow;
        j = maxCol;


        int query_matched_idx = 0;
        int target_matched_idx = 0;
        uint8_t result_query[MAX_SEQ_LEN];
        uint8_t result_target[MAX_SEQ_LEN];
        uint8_t direction;
        
        while (i > 0 && j > 0) {
            direction = directionMatrix[i][j];

            switch(direction) {
                case 0: // matched
                case 1: // mismatched
                    result_query[query_matched_idx++] = query_batch_line[j-1];
                    result_target[target_matched_idx++] = target_batch_line[i-1];
                    i--;
                    j--;
                break;
                case 2: // from upper cell
                    result_query[query_matched_idx++] = query_batch_line[j-1];
                    result_target[target_matched_idx++] = '-';
                    j--;
                break;
                case 3: // from left cell
                    result_query[query_matched_idx++] = '-';
                    result_target[target_matched_idx++] = target_batch_line[i-1];
                    i--;
                break;
            }
        }

        if (show_results) {
            std::cout << "Results[" << target_batch_idx << "]:" << std::endl;
            std::cout << "\tquery:\t";
            for (i = 0; i < query_matched_idx; i++) {
                std::cout << result_query[i];
            }
            std::cout << std::endl;

            std::cout << "\ttarget:\t";
            for (i = 0; i < target_matched_idx; i++) {
                std::cout << result_target[i];
            }
            std::cout << std::endl << std::endl;
        }
        target_batch_idx++;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Time elapsed: " << duration.count() << " ms" << std::endl;

    return 0;
}