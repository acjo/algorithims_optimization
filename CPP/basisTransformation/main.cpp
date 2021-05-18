#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#define CKRONECKER( x ) ( x == 0 ? 0 : 1)

using namespace std;

double metric( const vector<double>& vec1, const vector<double>& vec2, int norm= 2)
{
	const int largest = max(vec1.size(), vec2.size());

	double result = 0;

	for(auto i = 0; i < largest; ++i)
	{
		if (i >= vec1.size())
			result += pow(0 - vec2.at(i), norm);

		else if( i >= vec2.size())
			result += pow(vec1.at(i) - 0, norm);

		else
			result += pow(vec1.at(i) - vec2.at(i), norm);
	}

	return pow(result, 1.0/norm); 
}


int main( int argc, char* argv[] )
{

	vector<double> first = {1, 2, 3, 7};
	vector<double> second = {4, 5, 6};

	double distance = metric(first, second);

	cout << "distance: " << distance << endl;


	return 0;
}
