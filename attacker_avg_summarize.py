import matplotlib.pyplot as plt
import json
import sys


DBGD_output = open(sys.argv[1], 'r')
DBGD_output_lines = DBGD_output.readlines()
# Extract macro here
macro = json.loads(DBGD_output_lines[0])
# print(macro)

DBGD_output_lines = DBGD_output_lines[0:]

NDCG_attack = []
NDCG_label = []	
iterations = []
lr = []
# graph_title = ""
attack_name = ""

file_name = sys.argv[1]

if "new_freq_attack" in file_name:
	attack_name = "new_freq_attack"
elif "freq_attack" in file_name:
	attack_name = "freq_attack"
else:
	attack_name = "click_kth_doc"

for line in DBGD_output_lines:
	run_details = json.loads(line)['simulation_arguments']
	# graph_title = run_details['simulation_arguments']['click_models'][0]
	Tau = []
	num_clicks= []


	run_results = json.loads(line)['results']
	# print(graph_title)
	run_results = json.loads(line)['results']["NDCG_attack"][attack_name]["mean"]

	count = 0
	for val in run_results:
		if count > 0 and (count % 1000 == 0 or count == 9999):
			NDCG_attack.append(round(val,5))
			# count = 0
		count += 1


	run_results = json.loads(line)['results']["NDCG_label"][attack_name]["mean"]

	count = 0

	for val in run_results:
		if count % 1000 == 0:
			NDCG_label.append(val)
			count = 0
		count += 1

	run_results = json.loads(line)['results']["LR"][attack_name]["mean"]

	count = 0

	for val in run_results:

		if count > 0 and (count % 1000 == 0 or count == 9999):
			lr.append(round(val, 6))
			# count = 0

		count += 1



print("NDCG attack: ", NDCG_attack)
print("NDCG label: ", NDCG_label)
print("LR: ", lr)