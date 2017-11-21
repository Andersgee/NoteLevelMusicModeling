module CHECKPOINT

import JLD

function save_model(path, Wenc, benc, W, b, Wdec, bdec)
  JLD.save(path,
  "Wenc", Wenc,
  "benc",benc,
  "W", W,
  "b", b,
  "Wdec", Wdec,
  "bdec", bdec)
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

function save_optimizevars(path, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothvec, smoothvec2, smoothvec3)
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
  "smoothvec",smoothvec,
  "smoothvec2",smoothvec2,
  "smoothvec3",smoothvec3)

  println("saved ",path)
end

function load_optimizevars(path)
  d = JLD.load(path)
  mWenc = d["mWenc"]
  vWenc = d["vWenc"]
  mbenc = d["mbenc"]
  vbenc = d["vbenc"]
  Wm = d["Wm"]
  Wv = d["Wv"]
  bm = d["bm"]
  bv = d["bv"]
  mWdec = d["mWdec"]
  vWdec = d["vWdec"]
  mbdec = d["mbdec"]
  vbdec = d["vbdec"]
  gradientstep = d["gradientstep"]
  smoothvec = d["smoothvec"]
  smoothvec2 = d["smoothvec2"]
  smoothvec3 = d["smoothvec3"]

  println("loaded ",path)
  return mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothvec,smoothvec2,smoothvec3
end

end
