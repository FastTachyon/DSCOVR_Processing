# DSCOVR_Processing
Post-processing of publicly available data from the DSCOVR probe. 

## Context of the project
This project is our entry in the 2023 SpaceApps Hackathon. More precisely, we are trying to solve the challenge named "Develop the Oracle of DSCOVR". Our goal is to develop a program (or a workflow) which allows to roughly predict space weather indicators using data from the DSCOVR probe, which is known to be faulty form time to time.  

## Paticipants:
    Alexandre Beaulieu
    Mathieu Bergeron
    Samuel Fortin

## Data source
### Satellite products
We use L1 data from the [Deep Space Climate Observatory (DSCOVR)] probe located at the L1 lagrange point. The data was dowloaded directly from the challenge website. 

### Earth-based products
The space weather indicators are calculated using data from multiple sciences stations across the world. This data is compiled into indices which can be broadly interpreted by the [International Service of Geomagnetic Indices (ISGI)]. We requested the following indicators: aa, am, Kp, Dst, PC, AE, SC and SFE from january 1st, 2016  to the december 31, 2022. 
