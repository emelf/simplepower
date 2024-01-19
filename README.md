# Simplepower
Simplepower is a lightweight power flow tool. Its main functionallity is to solve the power flow equations using scipy.root(.). Version 0.0.1 have focused on solving the power flow equations and developing basic utilities like converting a grid with PV buses to a grid with only static generators and static loads. The current version, 0.1.0, plans on implementing quasi-dynamic simulation of the power system. This means loads can be modeled as ZIP, and generators can implement active- and reactive power droop. Plans for v0.1.0 are: 
- Add classes that enables Quasi-dynamic simulation (QDS)
- Add models that can emulate dynamic behavior in the QDS 
- Add controls, like active and reactive power droop in the QDS  