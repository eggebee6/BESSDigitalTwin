## Digital Twin for a Battery Energy Storage System

This repository contains work being done to create the intelligent agent for the digital twin (DT) of a battery energy storage system (BESS).  Details of the digital twin architecture and the BESS DT are available in *“The Use of Digital Twins in Inverter-based DERs to Improve Nanogrid Fault Recovery”* [1].

### Digital Twin
A digital twin has four layers:
- **Physical twin** - the physical device or system itself
- **Virtual twin** - a real-time, up-to-date digital replica of the physical twin
- **Intelligent agent** - a model of the learning and decision making processes
- **Data communications** - the exchange of information with other devices, systems, or operators

![DigitalTwin](https://user-images.githubusercontent.com/54823750/206030301-8659904a-4962-45bf-97ef-81286e3971ca.png)

The intelligent agent (or simply, 'the agent') contains the artificial intelligence and machine learning (AI/ML) features of the digital twin.  [MATLAB](https://www.mathworks.com/) and the [MATLAB Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/index.html) are being used to develop deep learning neural network models which will be implemented in the intelligent agent and deployed with the digital twin.

The agent analyzes the behavior of the physical twin through a physics-based model.  This model is made available by the virtual twin, which continually updates the model in real-time using measurements from the physical twin.  In particular, the virtual twin provides an error vector to the agent.  The error vector quantifies the difference between the received measurements and the measurements predicted by the model.  Analysis of the error vector will help the agent gain an understanding of the physical twin and its interaction with its environment.

The deep learing models being developed here are intended to explore various AI/ML applications to be used by digital twins in power systems, especially in the context of resilience (the ability to "deliver power where it's needed, when it's needed, even in degraded or damaged conditions").

### BESS DT for nanogrid resilience
For the BESS DT, the environment is the nanogrid to which the BESS is connected, and the goal of the agent is to recommend appropriate actions based on observations of local (i.e. at the BESS) voltage and current measurements alone.  For example, in response to a disturbance in the nanogrid, the agent could recommend whether the BESS should continue operating in grid-following mode or should switch to grid-forming mode.  The ability to recommend actions using only local observations reduces the reliance on a data communiation network coordinating multiple devices in the nanogrid.  Reduced reliance on the communication network improves nanogrid resilience since the BESS DT can take appropriate actions during simultaneous communication and electrical faults.

This project is based on a [Simulink](https://www.mathworks.com/products/simulink.html) simulation of a 3-phase AC industrial nanogrid.  The nanogrid contains the battery energy storage system, a solar photovoltaic (PV) farm, and a diesel generator along with three loads, one of which is an induction machine.  Simulation data is created by simulating the nanogrid into various load and generation configurations.  The simulation is saved in these states, then resumed later to simulate scenarios of 'interesting' events (a fault, a load step, etc.).  The dataset consists of BESS measurements and error vectors along with metadata including an event label.  The dataset is not currently available to the public, but another repository is under development which will contain the simulation model and scripts needed to generate the dataset.

### Miscellaneous
This project is being done with the [Center for Sustainable Electric Energy Systems](https://sites.uwm.edu/sees/) (CSEES) at the [University of Wisconsin - Milwaukee](https://uwm.edu/).  Additional support has been provided by the UWM [High Performance Computing](https://uwm.edu/hpc/) (HPC) group.

The CSEES GitHub page is available at https://github.com/UWM-SEES

Active development of this project should take place on the `dev` branch and should be compatible with MATLAB R2022a

A [wiki page](https://github.com/eggebee6/BESSDigitalTwin/wiki) is under construction to provide more information on this project.

---
### References
[1] A. Eggebeen, M. Vygoder, G. Oriti, J. Gudex, A. L. Julian and R. Cuzner, “The Use of Digital Twins in Inverter-based DERs to Improve Nanogrid Fault Recovery”, 2023 IEEE Energy Conversion Congress and Exposition (ECCE), Nashville, TN, Oct-Nov 2023.
