module CHECKPOINT

import JLD

function save_model(path, Wenc, benc, W, b, Wdec, bdec)
  JLD.save(path, "Wenc", Wenc, "benc", benc, "W", W, "b", b, "Wdec", Wdec, "bdec", bdec)
  println("saved ",path)
end

function load_model(path)
  d = JLD.load(path)
  Wenc = d["Wenc"]
  benc = d["benc"]
  W = d["W"]
  b = d["b"]
  Wdec = d["Wdec"]
  bdec = d["bdec"]
  println("loaded ",path)
  return Wenc, benc, W, b, Wdec, bdec
end

function save_optimizevars(path, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothcost)
  JLD.save(path,
  "mWenc",mWenc,
  "vWenc",vWenc,
  "mbenc",mbenc,
  "vbenc",vbenc,
  "Wm",Wm,
  "Wv",Wv,
  "bm",bm,
  "bv",bv,
  "mWdec",mWdec,
  "vWdec",vWdec,
  "mbdec",mbdec,
  "vbdec",vbdec,
  "gradientstep",gradientstep,
  "smoothcost",smoothcost)

  println("saved ",path)
end

function load_optimizevars(path)
  d = JLD.load(path)
  Wm = d["Wm"]
  Wv = d["Wv"]
  gradientstep = d["gradientstep"]
  smoothloss = d["smoothloss"]
  println("loaded ",path)
  return Wm, Wv, gradientstep, smoothloss
end

end
