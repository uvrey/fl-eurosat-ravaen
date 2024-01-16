# Federated learning for data classification and change detection on-board satellite constellations
![smaller banner](https://github.com/uvrey/l46-project/assets/77244149/e29122f1-d26c-47a3-944b-5bdfc6bbdf22)

## Overview
This project is presented in partial fulfilment of the requirements for L46: Principles of Machine Learning Systems. 

We explore several constraints associated with nanosatellite constellations by simulating a federated learning system on the EuroSAT dataset. In best conditions, an overall accuracy of 97.1\% was achieved, 1.7\% below the baseline. The most significant performance-affecting constraint is the iid-ness of the partitioned data, followed by the availability of concurrent clients. 

In the interest of investigating a real-world application of FL following this initial benchmarking process, we investigate the feasibility of federating RaVAEn, a SoTA system for unsupervised change detection on-board satellite constellations on the RAEVEN dataset for change detection from on-board satellite data.

The contributions of this work are as follows:

- We discuss three experiments to evaluate FL with the use case of classifying the EuroSAT dataset, which include exploring the effect of varying IID-ness and numbers of clients.
- We provide an initial `proof of concept' for the feasibility of federated learning on an aspect of the RaVAEn system (namely, a `SimpleVAE` training pipeline), employing a subset of the original project's associated dataset. 

## Set-up and Installation
The code may be compiled and executed for each element of this work as follows.

### EuroSAT


### RaVAEn
 Navigate to the `RaVAEn` folder and initialise its setup as follows:

```bash
# This will create a ravaen_env conda environment:
make requirements
conda activate ravaen_env
# Add these to open the prepared notebooks:
conda install nb_conda
jupyter notebook
# This will open an interactive notebook in your browser where you can navigate to the federated learning demo (fl_demo) and initialise this process. 
```


## Contributors
This project was developed by Luca Powell (LP647) and Josephine Rey (JMR239) as part of the MPhil in Advanced Computer Science programme, University of Cambridge, 2023. 

## Planning and Decision-Making
| Week | Date                | Actions   | Decisions |
| ------ | ------------------- | --------- | --------- |
| Week 1 | December 12 - 18    | Compile reading list, assess literature, propose project ideas.  | Project topic          |
| Week 2 | December 19 - 25    | Read literature, create Miro board for detailed planning, set up GitHub repo, Colab training environment and report outline.   | Task allocation and project milestones.   |
| Week 3 | December 26 - January 1 | Prepare literature review, introduction and abstract. Investigate RAEven model files and devise strategy for feeding to Flower. | Experiment selection and risk mitigation strategy |
| Week 4 | January 2 - 8       | Intensify development efforts, begin experiments. Iteratively report on findings and incorporate within report. | Revised experiment selection |
| Week 5 | January 9 - 15      | Complete experiments, finalise report, devise figures and proofread. | Submission pipeline. |

### Task Allocation
#### JR:
- **Report contributions**: Introduction, abstract and literature review
- Conduct and report on FedRAEven experiments
- Proof reading and editing
- Figures and tables

#### LP: 
- **Report contributions**:  Implementation and methods
- Implement FedRAEven

#### Shared tasks:
- Determine experiment list
- Training and federation set-up on machines
- **Report contributions:** Results, discussion, conclusion

