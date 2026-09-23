const fs=require("fs"),vm=require("vm"),assert=require("node:assert/strict"),path=require("path");
const root=path.resolve(__dirname,"..");
const files=["walking_overlay.js","overlay.js","app.js","api.js"];
for(const f of files)new vm.Script(fs.readFileSync(path.join(root,"backend/static/js",f),"utf8"),{filename:f});
const texts=[],images=[],contexts=[];
const ctx=new Proxy({measureText:t=>({width:t.length*7}),fillText:t=>texts.push(t),drawImage:()=>contexts.push("mask")},{get:(o,k)=>k in o?o[k]:(...a)=>{}});
const canvas={getContext:()=>ctx,clientWidth:400,clientHeight:800,width:400,height:800};
const video={videoWidth:540,videoHeight:960};
const win={devicePixelRatio:1,addEventListener:()=>{}};
const context=vm.createContext({window:win,document:{getElementById:id=>id==="overlay"?canvas:video},Image:class{constructor(){images.push(this)}},Math,Number});
for(const f of files.slice(0,2)){vm.runInContext(fs.readFileSync(path.join(root,"backend/static/js",f),"utf8"),context);Object.assign(context,win)}
const ev={risk_schema_version:1,image_width:540,image_height:960,mask_png:"example",roi:{corridor_polygons:[[[.2,.4],[.7,.4],[1,1]],[[.1,.4],[.3,.4],[.9,1]]],immediate_polygon:[[.2,.8],[.7,.8],[1,1]]},counts:{danger:1},level:"danger",warning_text:"위험 · 보행자 근접"};
win.GOverlay.draw([],ev);assert(texts.includes(ev.warning_text));assert.equal(images.length,1);
images[0].onload();assert.equal(contexts.length,1);assert(texts.filter(x=>x===ev.warning_text).length===2);
win.GOverlay.draw([],{...ev,warning_text:"새 경고"});
const old=images[1];win.GOverlay.clear();const count=texts.length;old.onload();assert.equal(texts.length,count);
win.GOverlay.setWalkingMaskEnabled(false);win.GOverlay.draw([],ev);assert.equal(images.length,2);assert(texts.includes(ev.warning_text));
video.videoWidth=960;video.videoHeight=540;const before=texts.length;win.GOverlay.draw([],ev);assert.equal(texts.length,before);
for(const state of ["red","green","unknown"]){
 const d=win.GOverlay.describeDetection({class_name:"pedestrian_signal",extra:{signal_state:state,color_confidence:.87}});
 assert.equal(d.label,{red:"빨간불",green:"초록불",unknown:"신호 미확인"}[state]);
 assert.equal(d.confidenceText,state==="unknown"?"":"87.0%");
}
console.log("PASS: JS syntax, canvas warning/mask composition, stale callbacks, rotation, toggle, traffic branch preservation");
