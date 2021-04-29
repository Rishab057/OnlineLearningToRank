import matplotlib.pyplot as plt
import json
import sys


DBGD_output = open(sys.argv[1], 'r')
print(sys.argv[1])
DBGD_output_lines = DBGD_output.readlines()
# Extract macro here
macro = json.loads(DBGD_output_lines[0])
# print(macro)

DBGD_output_lines = DBGD_output_lines[0:]

NDCG_attack_MQ2007 = []
NDCG_label_MQ2007 = []
LR_MQ2007 = []
NDCG_label = []	
iterations = []
graph_title = ""
attack_name = ""

file_name = sys.argv[1]

if "frequency" in file_name:
	attack_name = "frequency_attack"
else:
	attack_name = "naive_intersection_attack"

for line in DBGD_output_lines:
	run_details = json.loads(line)['simulation_arguments']
	graph_title = run_details['simulation_arguments']['attacker_click_model']
	Tau = []
	num_clicks= []


	run_results = json.loads(line)['results']
	print(graph_title)
	run_results = json.loads(line)['results']["NDCG_attack"][attack_name]["mean"]
	it = 0
	for val in run_results:
		NDCG_attack_MQ2007.append(val)


	run_results = json.loads(line)['results']["NDCG_label"][attack_name]["mean"]

	for val in run_results:
		NDCG_label.append(val)
		iterations.append(it)
		it += 1

	run_results = json.loads(line)['results']["NDCG_label"][attack_name]["mean"]

	for val in run_results:
		NDCG_label_MQ2007.append(val)

	run_results = json.loads(line)['results']["LR"][attack_name]["mean"]

	for val in run_results:
		LR_MQ2007.append(float('%.8f'%val))





fig_ndcg_attack, ax_ndcg_attack = plt.subplots()
ax_ndcg_attack.plot(iterations, NDCG_attack_MQ2007, '#FF0000', label="MQ2007 attacker")
ax_ndcg_attack.plot(iterations, NDCG_label_MQ2007, '#d79232', label="MQ2007 ground truth")
  

# ax_ndcg_attack.set_title("")
ax_ndcg_attack.set_xlabel("Iteration")
ax_ndcg_attack.set_ylabel("NDCG@10")
plt.legend()


fig_LR, ax_LR = plt.subplots()

ax_LR.plot(iterations, LR_MQ2007, '#FF0000', label="MQ2007")
# ax_LR.plot(iterations, LR_Yahoo, '#d79232', label="TD2003")
# ax_LR.plot(iterations, LR_MSLR, '#0000FF', label="MSLR")
# ax_LR.set_title(graph_title)
ax_LR.set_xlabel("Iteration")
ax_LR.set_ylabel("Learning Rate")
plt.legend()

plt.show()


# #FF0000  #800000
#d79232 #FFD700
#0000FF #00FFFF   

#DAA520
#636061
#FF6347