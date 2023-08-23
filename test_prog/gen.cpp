#include <fstream>
#include <string>
#include <sstream>

int main() {
	std::string line;
	std::ifstream infile("sequence3.fasta");
	std::getline(infile, line);
	std::getline(infile, line);
	std::ofstream outfile1("ref_319_20000.fasta");
	std::ofstream outfile2("query_319_20000.fasta");
	
	for (int i = 0; i < 20000; i++) {
		outfile1 << ">" << i+1 << std::endl;
		outfile1 << line << std::endl;
		outfile2 << ">" << i+1 << std::endl;
		outfile2 << line << std::endl;
	}
	outfile1.close();
	outfile2.close();
	return 0;
}
