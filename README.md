# Infection detection thanks to programmed bacteria
## The project
This project is part of an internship at IVS in Rikshospitalet, Oslo, Norway. 

The goal was to study the possibility of detecting a urinary tract infection (UTI) cause by UroPathogenic Esherichia Coli (UPEC) thanks to programmed commensal E.Coli. This work was based on the study [Computing within bacteria: Programming of bacterial behavior by means of a plasmid encoding a perceptron neural network A. Gargantilla Becerra, M. Guti ́errez, R. Lahoz-Beltra](https://pubmed.ncbi.nlm.nih.gov/35063580/).

This repository contains the simulation part of this project. The simulations were done using the software Gro, like in the original paper. 

## The simulation parameters
The model takes as input : 
- **AI-2 by UPEC** : a quorum sensing (QS) molecule used by commensal E.Coli and UPEC. The goal was to knowk-out the AI-2 producing genes in commensal E.Coli so that the only bacteria producing AI-2 are the ones causing the infection, UPEC. The programmed bacteria, commensal E.Coli, could detect this QS molecule and so detect the presence of UPEC
- **Urea/QS molecule** : At the beginning we wanted to use Urea as an input because it seems that Urea concentration is a factor risk for the development of a UTI with UPEC. But commensal E.Coli cannot sense Urea concentration so that would be very complicated, or impossible, to program the bacteria to sense this input. Instead, we wanted to use another quorum sensing molecule (not specified at the end of the project) that would be produced by commensal E.Coli (and possibly also UPEC) because it would give us information on how well the programmed bacteria is growing (and possibly UPEC). That would indeed give information on the envrionmental risk of infection. The fact that UPEC could also produce this QS molecule is not a problem because we also have information on how well UPEC is growing on its own thanks to AI-2, so we could access the information of how well commensal E.Coli grows with a difference or something like that.
- **Sidereophores** : siderophores are molecule produced by UPEC that allow them to scavange iron from the host environment. Iron is a very important element for bacterial growth, and it has been shown that the presence of siderophores is a very important virulence factor for UPEC. We found 4 different siderophores for UPEC : enterobactin, salmochelin, yersiniabactin, and aerobactin. "Siderophore" is a single input, considered present when at least one of the siderophores is present.
- **Toxins** : It has been shown that UPEC releases toxins, and that it is an important virulence factor for UPEC. Those toxins cause the disruption of immune response, tissue damage by lysis, by triggered apoptosis or cell exfoliation. That allows UPEC to better cross mucosal barriers, damage effector immune cells, and gain enhanced access to host nutrients and iron stores. Unfortunately, commensal E.Coli is not able to sense these toxins and trying to engineer them to do so would be way too complicated or impossible, so we abandonned this idea of input for the latest version of simulations.

## The files 
### InfectionDetection.gro
Latest gro script to simulate the infection detection. It 
