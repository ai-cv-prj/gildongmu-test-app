/* Walking review overlay: integration colors, observed geometry, shared warning text. */
window.GWalkingOverlay = (() => {
  const colors = { monitor: "#c8c8c8", caution: "#ffc800", danger: "#ff3434" };
  function draw(ctx, r, detections, ev) {
    ctx.save();
    ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
    const px = (p) => [r.x + p[0] * r.w, r.y + p[1] * r.h];
    function polygon(points, color, fill) {
      if (!points || points.length < 3) return;
      ctx.beginPath();
      points.forEach((p,i) => { const [x,y]=px(p); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
      ctx.closePath();
      ctx.fillStyle=fill; ctx.fill();
      ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
    }
    function label(text,x,y,color,size=11) {
      ctx.font=`600 ${size}px -apple-system, "Malgun Gothic", sans-serif`;
      ctx.textBaseline="top";
      let width=ctx.measureText(text).width+10;
      if (width>r.w-12) { size=Math.max(8,size*(r.w-12)/width); ctx.font=`600 ${size}px sans-serif`; width=ctx.measureText(text).width+10; }
      x=Math.max(r.x+4,Math.min(x,r.x+r.w-width-4));
      y=Math.max(r.y+4,Math.min(y,r.y+r.h-size-8));
      ctx.fillStyle="rgba(12,16,22,.88)"; ctx.fillRect(x,y,width,size+8);
      ctx.fillStyle=color; ctx.fillText(text,x+5,y+4);
    }
    const viewUnavailable = ev.camera_view?.status === "unavailable";
    for (const region of !viewUnavailable && ev.surface?.alert_level ? ev.surface.regions || [] : []) {
      polygon(region.polygon,"#ffd200","rgba(255,190,0,.20)");
    }
    const roi=ev.roi || {};
    if (!viewUnavailable) {
      for (const p of roi.corridor_polygons || [roi.corridor_polygon]) polygon(p,"#14dcff","rgba(20,220,255,.09)");
      polygon(roi.immediate_polygon,"#fa28dc","rgba(250,40,220,.14)");
    }
    const ordered=[...(detections||[])].sort((a,b)=>({monitor:0,caution:1,danger:2}[a.extra?.alert_level]||0)-({monitor:0,caution:1,danger:2}[b.extra?.alert_level]||0));
    for (const d of viewUnavailable ? [] : ordered) {
      const e=d.extra||{}, level=e.alert_level ?? e.risk_level;
      if (!level) continue;
      const color=colors[level]||colors.monitor;
      const [x1,y1]=px([d.box.x1,d.box.y1]), [x2,y2]=px([d.box.x2,d.box.y2]);
      ctx.strokeStyle=color; ctx.lineWidth=level==="monitor"?1:3; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      if (level!=="monitor" && e.warning_primary===false) continue;
      const state=e.alert_status==="uncertain" && level==="monitor" ? "UNKNOWN" : level.toUpperCase();
      const group=e.warning_group_size>1 ? ` x${e.warning_group_size}` : "";
      label(`${state} | ${e.display_label||d.class_name} ${d.track_id==null?"NEW":"#"+d.track_id}${group}`,x1,y1-22,color);
      if (level!=="monitor") {
        const reason=(e.reasons||[]).includes("side_close_candidate") ? "SIDE CLOSE" : (e.reasons||[]).includes("static_near_contact") ? "NEAR CONTACT" : e.in_path ? "PATH" : "CLOSE CANDIDATE";
        const ttc=Number.isFinite(e.ttc_s) ? `TTC~${e.ttc_s.toFixed(1)}s` : "TTC --";
        label(`${reason} | ${ttc}${e.hold_reason?" | HOLD":""}`,x1,y1+4,color,10);
      }
    }
    const c=ev.counts||{};
    label(`DANGER ${c.danger||0}   CAUTION ${(c.caution||0)+(c.surface||0)+(c.advisories||0)+(c.camera_view||0)}   MONITOR/UNK ${c.monitor||0}`,r.x+5,r.y+r.h-46,"#fff",11);
    if (viewUnavailable) label("CAMERA: 촬영 불가 · 전방을 비춰주세요",r.x+5,r.y+r.h-23,"#ccc",10);
    else label(`ROI: ${roi.source||"fixed"} · ${ev.tracker_status||"unknown"}${ev.state_reset?" · 상태 초기화":""}`,r.x+5,r.y+r.h-23,"#ccc",10);
    if (ev.warning_text) label(ev.warning_text,r.x+5,r.y+6,colors[ev.level]||colors.caution,Math.min(14,Math.max(11,r.w/27)));
    ctx.restore();
  }
  return {draw};
})();
