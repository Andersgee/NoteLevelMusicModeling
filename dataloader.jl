module DATALOADER

import NPZ
import Distributions

function get_batch!(batch, seqlen, bsz, data)
  lengths = [data[n][end,1] for n=1:length(data)]
  lengths = lengths.-seqlen #actually, subtract seqlen for correct weighted sampling
  for b=1:bsz
    #song = Distributions.wsample(1:length(data), lengths)

    # overfit a single song
    #song=3 # 3 means appass3 if beethoven
    song=2 # 2 means august if TchaikovskyPeter

    #perhaps... when constructing song here, shift everything by a random number [-3,.2...+2,+3]?
    #this will expand dataset (and only change key of a sampled song):
    #shift=rand(-3:3)
    shift=0

    X=zeros(256,data[song][end,1]) # construct (entire) manyhot from data[songnumber]
    for n=1:size(data[song],1)
        X[data[song][n,2]+shift, data[song][n,1]] = 1
    end

    s=rand(1:size(X,2)-seqlen) #random sequence within selected song
    batch[b]=X[:,s:s+seqlen] # (batch[b] length seqlen+1)
  end
end

function get_partialbatch!(x,t,batch,unrollsteps,batchstep,bsz)
  for b=1:bsz
    for i=1:unrollsteps
      x[i][:,b] .= batch[b][:,batchstep+i-1]
      t[i][:,b] .= batch[b][:,batchstep+i]
    end
  end
end

function load_dataset(lowerlimit)
print("\nLoading dataset... ")

data=[AlbenizIsac();
BachJohannSebastian();
BalakirewMiliAlexejewitsch();
BeethovenLudwigvan();
BorodinAlexander();
BrahmsJohannes();
BurgmuellerFriedrich();
ChopinFrederic();
ClementiMuzio();
DebussyClaude();
GodowskyLeopold();
GranadosEnrique();
GriegEdvard();
HaydnJoseph();
LisztFranz();
MendelssohnFelix();
MoszkowskiMoritz();
MozartWolfgangAmadeus();
MussorgskyModest();
RachmaninovSergey();
RavelMaurice();
ShubertFranz();
ShumannRobert();
SindingChristian();
TchaikovskyPeter()]

print("Done\n")

lengths = [data[n][end,1] for n=1:length(data)]
keep = lengths.>lowerlimit
data = data[keep]
lengths = [data[n][end,1] for n=1:length(data)]

println("(removed ", sum(.!keep), " songs for having length ", lowerlimit," or shorter)")
println("number of songs: ", sum(keep))
println("average length: ", mean(lengths))
println("shortest: ", minimum(lengths))
println("longest: ", maximum(lengths))
return data
end

function AlbenizIsac()
path="data/npy/AlbenizIsac/"
songs=["alb_esp1_format0.npy",
"alb_esp2_format0.npy",
"alb_esp3_format0.npy",
"alb_esp4_format0.npy",
"alb_esp5_format0.npy",
"alb_esp6_format0.npy",
"alb_se1_format0.npy",
"alb_se2_format0.npy",
"alb_se3_format0.npy",
"alb_se4_format0.npy",
"alb_se5_format0.npy",
"alb_se6_format0.npy",
"alb_se7_format0.npy",
"alb_se8_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function BachJohannSebastian()
path="data/npy/BachJohannSebastian/"
songs=["bach_846_format0.npy",
"bach_847_format0.npy",
"bach_850_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function BalakirewMiliAlexejewitsch()
path="data/npy/BalakirewMiliAlexejewitsch/"
songs=["islamei_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function BeethovenLudwigvan()
path="data/npy/BeethovenLudwigvan/"
songs=["appass_1_format0.npy",
"appass_2_format0.npy",
"appass_3_format0.npy",
"beethoven_hammerklavier_1_format0.npy",
"beethoven_hammerklavier_2_format0.npy",
"beethoven_hammerklavier_3_format0.npy",
"beethoven_hammerklavier_4_format0.npy",
"beethoven_les_adieux_1_format0.npy",
"beethoven_les_adieux_2_format0.npy",
"beethoven_les_adieux_3_format0.npy",
"beethoven_opus10_1_format0.npy",
"beethoven_opus10_2_format0.npy",
"beethoven_opus10_3_format0.npy",
"beethoven_opus22_1_format0.npy",
"beethoven_opus22_2_format0.npy",
"beethoven_opus22_3_format0.npy",
"beethoven_opus22_4_format0.npy",
"beethoven_opus90_1_format0.npy",
"beethoven_opus90_2_format0.npy",
"elise_format0.npy",
"mond_1_format0.npy",
"mond_2_format0.npy",
"mond_3_format0.npy",
"pathetique_1_format0.npy",
"pathetique_2_format0.npy",
"pathetique_3_format0.npy",
"waldstein_1_format0.npy",
"waldstein_2_format0.npy",
"waldstein_3_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function BorodinAlexander()
path="data/npy/BorodinAlexander/"
songs=["bor_ps1_format0.npy",
"bor_ps2_format0.npy",
"bor_ps3_format0.npy",
"bor_ps6_format0.npy",
"bor_ps7_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function BrahmsJohannes()
path="data/npy/BrahmsJohannes/"
songs=["brahms_opus117_1_format0.npy",
"brahms_opus117_2_format0.npy",
"brahms_opus1_1_format0.npy",
"brahms_opus1_2_format0.npy",
"brahms_opus1_3_format0.npy",
"brahms_opus1_4_format0.npy",
"br_im2_format0.npy",
"br_im5_format0.npy",
"br_im6_format0.npy",
"br_rhap_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function BurgmuellerFriedrich()
path="data/npy/BurgmuellerFriedrich/"
songs=["burg_erwachen_format0.npy",
"burg_geschwindigkeit_format0.npy",
"burg_gewitter_format0.npy",
"burg_perlen_format0.npy",
"burg_quelle_format0.npy",
"burg_spinnerlied_format0.npy",
"burg_sylphen_format0.npy",
"burg_trennung_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function ChopinFrederic()
path="data/npy/ChopinFrederic/"
songs=["chpn_op10_e01_format0.npy",
"chpn_op10_e05_format0.npy",
"chpn_op10_e12_format0.npy",
"chpn_op23_format0.npy",
"chpn_op25_e11_format0.npy",
"chpn_op25_e12_format0.npy",
"chpn_op25_e1_format0.npy",
"chpn_op25_e2_format0.npy",
"chpn_op25_e3_format0.npy",
"chpn_op25_e4_format0.npy",
"chpn_op27_1_format0.npy",
"chpn_op27_2_format0.npy",
"chpn_op33_2_format0.npy",
"chpn_op33_4_format0.npy",
"chpn_op35_1_format0.npy",
"chpn_op35_2_format0.npy",
"chpn_op35_3_format0.npy",
"chpn_op35_4_format0.npy",
"chpn_op53_format0.npy",
"chpn_op66_format0.npy",
"chpn_op7_1_format0.npy",
"chpn_op7_2_format0.npy",
"chpn-p10_format0.npy", #205
"chpn-p11_format0.npy", #163
"chpn-p12_format0.npy",
"chpn-p13_format0.npy",
"chpn-p14_format0.npy", #222
"chpn-p15_format0.npy",
"chpn-p16_format0.npy",
"chpn-p17_format0.npy",
"chpn-p18_format0.npy",
"chpn-p19_format0.npy",
"chpn-p1_format0.npy", #207
"chpn-p20_format0.npy", #73
"chpn-p21_format0.npy",
"chpn-p22_format0.npy", #297
"chpn-p23_format0.npy",
"chpn-p24_format0.npy",
"chpn-p2_format0.npy", #191
"chpn-p3_format0.npy",
"chpn-p4_format0.npy",
"chpn-p5_format0.npy", #282
"chpn-p6_format0.npy", #196
"chpn-p7_format0.npy", #68
"chpn-p8_format0.npy",
"chpn-p9_format0.npy", #215
"chp_op18_format0.npy",
"chp_op31_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function ClementiMuzio()
path="data/npy/ClementiMuzio/"
songs=["clementi_opus36_1_1_format0.npy",
"clementi_opus36_1_2_format0.npy", #294
"clementi_opus36_1_3_format0.npy",
"clementi_opus36_2_1_format0.npy",
"clementi_opus36_2_2_format0.npy", #162
"clementi_opus36_2_3_format0.npy",
"clementi_opus36_3_1_format0.npy",
"clementi_opus36_3_2_format0.npy", #201
"clementi_opus36_3_3_format0.npy",
"clementi_opus36_4_1_format0.npy",
"clementi_opus36_4_2_format0.npy", #296
"clementi_opus36_4_3_format0.npy",
"clementi_opus36_5_1_format0.npy",
"clementi_opus36_5_2_format0.npy",
"clementi_opus36_5_3_format0.npy",
"clementi_opus36_6_1_format0.npy",
"clementi_opus36_6_2_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function DebussyClaude()
path="data/npy/DebussyClaude/"
songs=["deb_clai_format0.npy",
"deb_menu_format0.npy",
"deb_pass_format0.npy",
"deb_prel_format0.npy",
"debussy_cc_1_format0.npy",
"debussy_cc_2_format0.npy",
"debussy_cc_3_format0.npy",
"debussy_cc_4_format0.npy",
"debussy_cc_5_format0.npy", #246
"debussy_cc_6_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function GodowskyLeopold()
path="data/npy/GodowskyLeopold/"
songs=["god_alb_esp2_format0.npy",
"god_chpn_op10_e01_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function GranadosEnrique()
path="data/npy/GranadosEnrique/"
songs=["gra_esp_2_format0.npy",
"gra_esp_3_format0.npy",
"gra_esp_4_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function GriegEdvard()
path="data/npy/GriegEdvard/"
songs=["grieg_album_format0.npy",
"grieg_berceuse_format0.npy",
"grieg_brooklet_format0.npy",
"grieg_butterfly_format0.npy",
"grieg_elfentanz_format0.npy",
"grieg_halling_format0.npy", #243
"grieg_kobold_format0.npy",
"grieg_march_format0.npy",
"grieg_once_upon_a_time_format0.npy",
"grieg_spring_format0.npy",
"grieg_voeglein_format0.npy",
"grieg_waechter_format0.npy",
"grieg_walzer_format0.npy",
"grieg_wanderer_format0.npy", #189
"grieg_wedding_format0.npy",
"grieg_zwerge_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function HaydnJoseph()
path="data/npy/HaydnJoseph/"
songs=["hay_40_1_format0.npy",
"hay_40_2_format0.npy",
"haydn_33_1_format0.npy",
"haydn_33_2_format0.npy",
"haydn_33_3_format0.npy",
"haydn_35_1_format0.npy",
"haydn_35_2_format0.npy",
"haydn_35_3_format0.npy",
"haydn_43_1_format0.npy",
"haydn_43_2_format0.npy",
"haydn_43_3_format0.npy",
"haydn_7_1_format0.npy",
"haydn_7_2_format0.npy",
"haydn_7_3_format0.npy",
"haydn_8_1_format0.npy",
"haydn_8_2_format0.npy", #233
"haydn_8_3_format0.npy", #294
"haydn_8_4_format0.npy", #283
"haydn_9_1_format0.npy",
"haydn_9_2_format0.npy",
"haydn_9_3_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function LisztFranz()
path="data/npy/LisztFranz/"
songs=["liz_donjuan_format0.npy",
"liz_et1_format0.npy",
"liz_et2_format0.npy",
"liz_et3_format0.npy",
"liz_et4_format0.npy",
"liz_et5_format0.npy",
"liz_et6_format0.npy",
"liz_et_trans4_format0.npy",
"liz_et_trans5_format0.npy",
"liz_et_trans8_format0.npy",
"liz_liebestraum_format0.npy",
"liz_rhap02_format0.npy",
"liz_rhap09_format0.npy",
"liz_rhap10_format0.npy",
"liz_rhap12_format0.npy",
"liz_rhap15_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function MendelssohnFelix()
path="data/npy/MendelssohnFelix/"
songs=["mendel_op19_1_format0.npy",
"mendel_op19_2_format0.npy",
"mendel_op19_3_format0.npy",
"mendel_op19_4_format0.npy",
"mendel_op19_5_format0.npy",
"mendel_op19_6_format0.npy",
"mendel_op30_1_format0.npy",
"mendel_op30_2_format0.npy",
"mendel_op30_3_format0.npy", #228
"mendel_op30_4_format0.npy",
"mendel_op30_5_format0.npy",
"mendel_op53_5_format0.npy",
"mendel_op62_3_format0.npy",
"mendel_op62_4_format0.npy", #234
"mendel_op62_5_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function MoszkowskiMoritz()
path="data/npy/MoszkowskiMoritz/"
songs=["mos_op36_6_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function MozartWolfgangAmadeus()
path="data/npy/MozartWolfgangAmadeus/"
songs=["mz_311_1_format0.npy",
"mz_311_2_format0.npy",
"mz_311_3_format0.npy",
"mz_330_1_format0.npy",
"mz_330_2_format0.npy",
"mz_330_3_format0.npy",
"mz_331_1_format0.npy",
"mz_331_2_format0.npy",
"mz_331_3_format0.npy",
"mz_332_1_format0.npy",
"mz_332_2_format0.npy",
"mz_332_3_format0.npy",
"mz_333_1_format0.npy",
"mz_333_2_format0.npy",
"mz_333_3_format0.npy",
"mz_545_1_format0.npy",
"mz_545_2_format0.npy",
"mz_545_3_format0.npy",
"mz_570_1_format0.npy",
"mz_570_2_format0.npy",
"mz_570_3_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function MussorgskyModest()
path="data/npy/MussorgskyModest/"
songs=["muss_1_format0.npy",
"muss_2_format0.npy",
"muss_3_format0.npy",
"muss_4_format0.npy", #260
"muss_5_format0.npy",
"muss_6_format0.npy",
"muss_7_format0.npy",
"muss_8_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function RachmaninovSergey()
path="data/npy/RachmaninovSergey/"
songs=["rac_op23_2_format0.npy",
"rac_op23_3_format0.npy",
"rac_op23_5_format0.npy",
"rac_op23_7_format0.npy",
"rac_op32_13_format0.npy",
"rac_op32_1_format0.npy",
"rac_op3_2_format0.npy",
"rac_op33_5_format0.npy",
"rac_op33_6_format0.npy",
"rac_op33_8_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function RavelMaurice()
path="data/npy/RavelMaurice/"
songs=["ravel_miroirs_1_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function ShubertFranz()
path="data/npy/ShubertFranz/"
songs=["schu_143_1_format0.npy",
"schu_143_2_format0.npy",
"schu_143_3_format0.npy",
"schub_d760_1_format0.npy",
"schub_d760_2_format0.npy",
"schub_d760_3_format0.npy",
"schub_d760_4_format0.npy",
"schub_d960_1_format0.npy",
"schub_d960_2_format0.npy",
"schub_d960_3_format0.npy",
"schub_d960_4_format0.npy",
"schubert_D850_1_format0.npy",
"schubert_D850_2_format0.npy",
"schubert_D850_3_format0.npy",
"schubert_D850_4_format0.npy",
"schubert_D935_1_format0.npy",
"schubert_D935_2_format0.npy",
"schubert_D935_3_format0.npy",
"schubert_D935_4_format0.npy",
"schuim-1_format0.npy",
"schuim-2_format0.npy",
"schuim-3_format0.npy",
"schuim-4_format0.npy",
"schumm-1_format0.npy",
"schumm-2_format0.npy",
"schumm-3_format0.npy",
"schumm-4_format0.npy",
"schumm-5_format0.npy",
"schumm-6_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function ShumannRobert()
path="data/npy/ShumannRobert/"
songs=["schum_abegg_format0.npy",
"scn15_10_format0.npy", #220
"scn15_11_format0.npy",
"scn15_12_format0.npy", #224
"scn15_13_format0.npy", #148
"scn15_1_format0.npy",
"scn15_2_format0.npy",
"scn15_3_format0.npy",
"scn15_4_format0.npy", #137
"scn15_5_format0.npy",
"scn15_6_format0.npy", #245
"scn15_7_format0.npy", #266
"scn15_8_format0.npy", #226
"scn15_9_format0.npy",
"scn16_1_format0.npy",
"scn16_2_format0.npy",
"scn16_3_format0.npy",
"scn16_4_format0.npy",
"scn16_5_format0.npy",
"scn16_6_format0.npy",
"scn16_7_format0.npy",
"scn16_8_format0.npy",
"scn68_10_format0.npy", #205
"scn68_12_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function SindingChristian()
path="data/npy/SindingChristian/"
songs=["fruehlingsrauschen_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

function TchaikovskyPeter()
path="data/npy/TchaikovskyPeter/"
songs=["ty_april_format0.npy",
"ty_august_format0.npy",
"ty_dezember_format0.npy",
"ty_februar_format0.npy",
"ty_januar_format0.npy",
"ty_juli_format0.npy",
"ty_juni_format0.npy",
"ty_maerz_format0.npy",
"ty_mai_format0.npy",
"ty_november_format0.npy",
"ty_oktober_format0.npy",
"ty_september_format0.npy"]
data=[NPZ.npzread(string(path,song)) for song in songs]
return data
end

end