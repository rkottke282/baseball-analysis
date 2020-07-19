## The Data
- because the datasets are so large, the raw data is not in git.
- you'll have to run this (locally) once to initialize the small dataset that 
  the analysis.py runs on.
- to bring in the raw data, create a raw_data directory at the root level
  and populate the directory and add the files to be read in.  
    - 2019_atbats.csv, atbats.csv
    - 2019_games.csv, games.csv
    - 2019_pitches.csv, pitches.csv
    - players_names.csv
- the data.py file contains helpers to clean the data and subset it.  See in analysis.py
    where a subset of data is stored in 'ss_data.csv'.
- after this subset of data has been created on a locally run, 
    the data load step will not occur, so subsequent runs should be more efficient
- the data file will have the following attributes (specifically for one player):
        inning: inning of the game
        outs: number of outs at pitch
        p_score: pitchers score at pitch
        b_score: batters score at pitch
        stand: batters stance converted to int (L=1.0 or R=0.0)
        top: boolean translated to int, top of the inning or not (True=1.0 or False=0.0)
        b_count: count of balls at pitch
        s_count: count of strikes at pitch
        pitch_num: number of pitches at pitch
        on_1b: number of players on first base (0 or 1)
        on_2b: number of players on second base (0 or 1)
        on_3b: number of players on third base (0 or 1)
        type: result of pitch (see appendix)
        pitch_type: type of pitch (see appendix)
        pitch_class: fastball=0.0, off-speed=1.0 (see appendix)

### Appendix

Type:
B - Ball
*B - Ball in dirt
S - Swinging Strike
C - Called Strike
F - Foul
T - Foul Tip
L - Foul Bunt
I - Intentional Ball
W - Swinging Strike (Blocked)
M - Missed Bunt
P - Pitchout
Q - Swinging pitchout
R - Foul pitchout
Values that only occur on last pitch of at-bat:
X - In play, out(s)
D - In play, no out
E - In play, runs
H - Hit by pitch

pitch_type: type of pitch
AB - ??
CH - Changeup [Off-Speed]
CU - Curveball [Off-Speed]
EP - Eephus* [Off-Speed]
FA - Four-seam Fastball [Fastball]
FC - Fastball Cutter [Fastball]
FF - Four-seam Fastball [Fastball]
FO - Pitchout (also PO)*
FS - Fastball sinker (similar to SI/SF) [Fastball]
FT - Two-seam Fastball [Fastball]
IN - Intentional ball
KC - Knuckle curve [Off-Speed]
KN - Knuckeball [Off-Speed]
PO - Pitchout (also FO)*
SC - Screwball* (opposite of curve) [Off-Speed]
SI - Sinker (similar to FS/SF) [Fastball]
SL - Slider [Off-Speed]
UN - Unknown*
