document.addEventListener("DOMContentLoaded",()=>{
  const menuBtn=document.getElementById("menuBtn"); if(menuBtn) menuBtn.addEventListener("click",()=>document.body.classList.toggle("menu-open"));
  const mobileMenuBtn=document.getElementById("mobileMenuBtn"); if(mobileMenuBtn) mobileMenuBtn.addEventListener("click",()=>document.body.classList.toggle("menu-open"));
  document.querySelectorAll(".sidebar a").forEach(a=>a.addEventListener("click",()=>document.body.classList.remove("menu-open")));
  const liteBtn=document.getElementById("liteModeBtn"); const applyLite=(v)=>{document.body.classList.toggle("lite-mode",v)}; applyLite(localStorage.getItem("mannina-maga-lite")==="1"); if(liteBtn) liteBtn.addEventListener("click",()=>{const v=!document.body.classList.contains("lite-mode");localStorage.setItem("mannina-maga-lite",v?"1":"0");applyLite(v)});
  const connection=document.getElementById("connectionText"); const updateConnection=()=>{if(connection) connection.textContent=navigator.onLine?"Online":"Offline"}; window.addEventListener("online",updateConnection);window.addEventListener("offline",updateConnection);updateConnection();
  if("serviceWorker" in navigator) navigator.serviceWorker.register("/service-worker.js").catch(()=>{});

  const liveBtn=document.getElementById("liveWeatherBtn");
  if(liveBtn) liveBtn.addEventListener("click",async()=>{
    const location=liveBtn.dataset.location||window.cropPilotLocation||"";
    if(!location){alert("Please add your village/city in your farm profile first.");return}
    const old=liveBtn.textContent; liveBtn.disabled=true; liveBtn.textContent="Loading weather...";
    try{const r=await fetch(`/api/weather?location=${encodeURIComponent(location)}`);const d=await r.json();if(!r.ok) throw new Error(d.error||"Weather unavailable");
      document.getElementById("temp").value=d.temperature ?? ""; document.getElementById("humidity").value=d.humidity ?? "";
      // Current daily rain is not the same as full-season rainfall. Keep rainfall field unchanged unless it is empty.
      const rain=document.getElementById("rainfall"); if(rain && !rain.value) rain.value=d.rainfall_today ?? "";
      liveBtn.textContent=`✓ ${d.location}: ${d.temperature}°C, ${d.humidity}%`;
    }catch(e){alert("Live weather could not be loaded. Please check your internet and location.");liveBtn.textContent=old}finally{liveBtn.disabled=false}
  });

  const canvas=document.getElementById("trendChart"); if(canvas&&window.analyticsData){
    const ctx=canvas.getContext("2d"),dpr=window.devicePixelRatio||1,width=canvas.clientWidth,height=360;canvas.width=width*dpr;canvas.height=height*dpr;ctx.scale(dpr,dpr);
    const pad={l:45,r:18,t:25,b:42},w=width-pad.l-pad.r,h=height-pad.t-pad.b,months=window.analyticsData.months,series=window.analyticsData.series,colors=["#16a34a","#0ea5e9","#f59e0b"],minY=1.8,maxY=3.7;
    ctx.font="12px system-ui";ctx.strokeStyle="#e2e8f0";ctx.fillStyle="#64748b";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=pad.t+(h/4)*i;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+w,y);ctx.stroke();ctx.fillText((maxY-(maxY-minY)*(i/4)).toFixed(1),8,y+4)}
    months.forEach((m,i)=>{const x=pad.l+(w/(months.length-1))*i;ctx.fillText(m,x-10,height-15)});
    Object.keys(series).forEach((name,idx)=>{ctx.strokeStyle=colors[idx];ctx.fillStyle=colors[idx];ctx.lineWidth=3;ctx.beginPath();series[name].forEach((v,i)=>{const x=pad.l+(w/(months.length-1))*i,y=pad.t+h-((v-minY)/(maxY-minY))*h;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();series[name].forEach((v,i)=>{const x=pad.l+(w/(months.length-1))*i,y=pad.t+h-((v-minY)/(maxY-minY))*h;ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill()})});
  }
});
