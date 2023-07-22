import numpy as np
import matplotlib.pyplot as plt

q_new_slot_4ms = [431, 483, 520, 533, 547, 567, 570, 572, 580, 593, 606, 602, 618, 634, 640, 636, 620, 649, 639, 637, 633, 626, 633, 650, 645, 657, 656, 631, 642, 646, 645, 631, 644, 640, 652, 651, 654, 646, 645, 663, 651, 656, 653, 662, 658, 666, 659, 670, 649, 644, 659, 650, 657, 672, 668, 670, 658, 669, 660, 692, 642, 669, 658, 666, 663, 679, 665, 673, 695, 659, 686, 672, 670, 677, 675, 682, 679, 669, 661, 648, 649, 681, 675, 669, 662, 665, 680, 691, 669, 669, 670, 677, 676, 669, 660, 669, 648, 678, 668, 679, 666, 693, 677, 661, 665, 683, 674, 656, 678, 682, 662, 651, 666, 683, 680, 651, 665, 682, 683, 687, 676, 674, 660, 659, 677, 680, 679, 671, 672, 661, 663, 670, 683, 660, 676, 671, 654, 675, 667, 672, 657, 674, 659, 662, 658, 668, 679, 682, 689, 689, 654, 655, 661, 680, 669, 669, 669, 682, 666, 675, 675, 665, 681, 676, 670, 670, 686, 670, 673, 663, 665, 663, 666, 667, 677, 675, 670, 672, 672, 667, 675, 678, 691, 686, 666, 656, 695, 672, 664, 664, 672, 680, 653, 665, 695, 671, 660, 663, 671, 666, 670, 686, 674, 668, 684, 665, 661, 685, 680, 663, 657, 680, 670, 655, 680, 681, 682, 664, 681, 676, 680, 672, 653, 694, 669, 675, 669, 651, 658, 680, 684, 674, 671, 680, 680, 670, 666, 667, 676, 669, 680, 684, 685, 678, 670, 664, 670, 654, 671, 663, 642, 685, 677, 673, 669, 673, 679, 669, 667, 655, 684, 675, 680, 692, 690, 687, 675, 660, 670, 663, 680, 670, 662, 663, 676, 673, 672, 675, 665, 687, 687, 677, 678, 673, 673, 672, 666, 682, 684, 672, 671, 690, 662, 677, 674, 661, 687, 676, 680, 681, 679, 684, 682, 674, 661, 680, 660, 654, 678, 681, 665, 674, 680, 678, 664, 688, 679, 697, 654, 680, 678, 678, 669, 676, 661, 671, 671, 665, 674, 688, 687, 676, 692, 679, 678, 671, 693, 670, 677, 660, 695, 657, 679, 677, 665, 682, 689, 686, 666, 666, 669, 652, 665, 674, 677, 689, 701, 668, 656, 686, 680, 684, 659, 679, 680, 666, 662, 657, 664, 665, 666, 669, 662, 677, 662, 699, 678, 673, 698, 667, 683, 680, 664, 667, 659, 690, 700, 684, 689, 670, 684, 689, 679, 661, 648, 668, 670, 661, 687, 674, 678, 674, 684, 679, 683, 677, 676, 670, 674, 689, 681, 670, 670, 679, 688, 674, 683, 674, 673, 673, 676, 659, 673, 661, 673, 696, 691, 683, 687, 696, 678, 681, 672, 690, 673, 665, 671, 691, 694, 674, 671, 679, 671, 690, 684, 695, 703, 680, 678, 687, 668, 695, 680, 691, 666, 671, 692, 686, 675, 648, 676, 682, 667, 684, 669, 679, 677, 681, 662, 681, 688, 693, 671, 670, 666, 672, 688, 694, 672, 663, 675, 691, 680, 674, 696, 650, 696, 688, 677, 678, 682, 672, 677, 655, 662, 672, 676, 672, 686, 684]
expanded_data = [688] * 1000

for i in range(500):
    expanded_data[2*i] = q_new_slot_4ms[i]
    if i != 499:
        expanded_data[2*i+1] = int((q_new_slot_4ms[i] + q_new_slot_4ms[i+1]) / 2)


for i in range(1000):
    expanded_data[i] = int(1.15*expanded_data[i])
plt.plot(expanded_data)
plt.show()
print(expanded_data)
