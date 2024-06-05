# Infection detection thanks to programmed bacteria
## The project
This project is part of an internship at IVS in Rikshospitalet, Oslo, Norway. 

The goal was to study the possibility of detecting a urinary tract infection (UTI) cause by UroPathogenic Esherichia Coli (UPEC) thanks to programmed commensal E.Coli. This work was based on the study [Computing within bacteria: Programming of bacterial behavior by means of a plasmid encoding a perceptron neural network A. Gargantilla Becerra, M. Guti ́errez, R. Lahoz-Beltra](https://pubmed.ncbi.nlm.nih.gov/35063580/).

This repository contains the simulation part of this project. The simulations were done using the software Gro, like in the original paper. 

## The simulation parameters
<img width="687" alt="Inputs-outputs" src="https://github.com/Selena-JM/BacteriaBehaviorProgramming_Gro_en/assets/160735287/294d0f9f-e06d-42e7-b39d-68803bccb1bc">

The model takes as input : 

- ***AI-2 by UPEC*** : a quorum sensing (QS) molecule used by commensal E.Coli and UPEC. The goal was to knowk-out the AI-2 producing genes in commensal E.Coli so that the only bacteria producing AI-2 are the ones causing the infection, UPEC. The programmed bacteria, commensal E.Coli, could detect this QS molecule and so detect the presence of UPEC
- ***Urea/QS molecule*** : At the beginning we wanted to use Urea as an input because it seems that Urea concentration is a factor risk for the development of a UTI with UPEC. But commensal E.Coli cannot sense Urea concentration so that would be very complicated, or impossible, to program the bacteria to sense this input. Instead, we wanted to use another quorum sensing molecule (not specified at the end of the project) that would be produced by commensal E.Coli (and possibly also UPEC) because it would give us information on how well the programmed bacteria is growing (and possibly UPEC). That would indeed give information on the envrionmental risk of infection. The fact that UPEC could also produce this QS molecule is not a problem because we also have information on how well UPEC is growing on its own thanks to AI-2, so we could access the information of how well commensal E.Coli grows with a difference or something like that.
- ***Sidereophores*** : siderophores are molecule produced by UPEC that allow them to scavange iron from the host environment. Iron is a very important element for bacterial growth, and it has been shown that the presence of siderophores is a very important virulence factor for UPEC. We found 4 different siderophores for UPEC : enterobactin, salmochelin, yersiniabactin, and aerobactin. "Siderophore" is a single input, considered present when at least one of the siderophores is present.
- ***Toxins*** : It has been shown that UPEC releases toxins, and that it is an important virulence factor for UPEC. Those toxins cause the disruption of immune response, tissue damage by lysis, by triggered apoptosis or cell exfoliation. That allows UPEC to better cross mucosal barriers, damage effector immune cells, and gain enhanced access to host nutrients and iron stores. Unfortunately, commensal E.Coli is not able to sense these toxins and trying to engineer them to do so would be way too complicated or impossible, so we abandonned this idea of input for the latest version of simulations.

The model gives as output a risk factor. As a first approach, this risk factor is simply the number of risk factor detected and it is shown by a color of fluorescence. 
- No risk factor detected : Risk = 0, no fluorescence
- 1 risk factor detected : Risk = 1, green fluorescence
- 2 risk factors detected : Risk = 2, blue fluorescence
- 3 risk factors detected : Risk = 3, yellow fluorescence
- 4 risk factors detected : Risk = 4, red fluorescence

The idea of the project was to try to program the bacteria with neural networks as was mentionned in the original paper. In the latter, they did not really use neural networks, they used a truth table to associate the input detection to a specific output. We wanted to see if it was possible to train a neural network to associate the presence of certain inputs to a certain risk. But that was not possible due to : 
- Lack of data : we have no data that associates those specific types of inputs to an infection risk. I tried to see if it was possible to synthesize data but that was not possible because of the second issue
- We cannot work with non binary inputs : trying to influence the way the bacteria reacts to the sensing of something is a huge work in itself. Bacteria mostly sense things binarily : the detection of an input triggers different reactions, and we cannot engineer the thresholds, we have to work with how th bacteria works, so we had to work with binary inputs. So there were only 16 different input combinations, which means we don't have to use any kind of machine learning, we juste have to create a truth table ourselves, like in the original paper. 

Nevertheless, I kept in this repository the different scripts I had written for data synthesis and machine learning so that they may be used for different purposes.

## The files
### Old_InfectionDetection.gro
First simulations I did. I changed the scripts given in the original paper to fit our project. In this script there are 4 different inputs : AI-2, Urea, Siderophores, Toxins. And the risk in output is the number of risk factor detected. 

This script executes a predefined simulation during which the risk factors increase each at a time so that we can see the bacteria going from no fluorescence, to green, to blue, to yellow, then to red and then all the risk factors disapear to show that the bacteria go back to no fluorescence when they detect nothing.

### InfectionDetection.gro
Latest gro script, where the inputs are AI-2, QS, Siderophores, Toxins. One can choose whether to include toxins as an input with the parameter 'number_inputs' : if set to 4 then all the inputs are kepts, if set to 3 then the toxins are not considered as an input since they are not possible to include in a lab experiment. Only 3 and 4 are accepted values for 'number_inputs'. 

The risk in output is the number of risk factor detected, as in the previous script Old_InfectionDetection.gro. In this script, the input-output association is more effecient since there is no need to write the whole truth table : the sum of detected inputs is computed and the adequate fluorescence is shown.

As in Old_InfectionDetection.gro, there is a specific script that is executed to show all the different possibilities of fluorescence.

### NeuralNetwork.py
The script shows all the different methods of machine learning we could have used :
- Solving linear equations
- Linear regression
- Decision trees
- Neural Network -> this was not finished because it did not make sense to continue working on this but I included it because it is what had been done in the original paper

All the methods listed above are either used with categories (the ouptput is juste a number in [0;4] to signify the risk category), or classifications (the ouput is a 1x5 vector, with [1,0,0,0,0] being the risk O, [0,1,0,0,0] being risk 1 etc

In the first part "linear regression" I also tried to synthesize data using a linear regression : each variable is linearly created with respect to the risk, with a director coefficient of 4. This method is not viable because each variable can be used alone for the classification : if urea is greater than 0.85 for example it means it is a risk 4, which is false 

### Synthesize_data.py
I wrote this script using [The Synthetic Data Vault script](https://colab.research.google.com/drive/1F3WWduNjcX4oKck6XkjlwZ9zIsWlTGEM) for in case we could stop considering binary inputs. We could have used this script to synthetize data : 
    - Create some real data with the real concentrations and we assign an infection risk
    - Use this adapted and functionnal script to synthetize more data
    - Train a more complicated NN than a single layer perceptron

### decisionTreeRules.txt
This document explains the decision tree rules for risk classification with 2 visualisations, for 2 cases. Those are the results of the decision tree method in 2 cases : the naive case (output is the number of risk factors detected), and the linear case (the data is created linearly to the risk factor).

