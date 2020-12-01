In this paper, we adopt three datasets as follows:

	1) 	America_data.txt
	   	America_impact.mat

	2) 	Australia_data.txt
		Australia_impact.mat
		Australia_original_data.mat

	3)	UK_data.txt
		UK_impact.mat

How to run experiments:

	1) Comparison against the State-of-the-art Competitors
		run hashing.py with "_data.txt" and "_impact.mat" to get "mapes" in "*_results.mat" as MAPEs on three datasets for Table 1

		run visualization.m with "Australia_original_data.mat" to get "Y" as Dimension 1 and Dimension 2, and labels as Resilience for Figure 3 

		run daily_ape.m with "*_results.mat" to get "temporal_ape" as Daily APEs on three datasets for Figure 4

	2) Keyword Understanding
		run hashing_word.py with "_data.txt" and "_impact.mat" to get the statistics of the keywords, "keywords" in "*_hashing_word.mat"
		
		run keywords.m with "*_hashing_word.mat" to get "temporal_weight" as the temporal evolution of the keywords for Figure 5


