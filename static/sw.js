const CACHE="mannina-maga-farmer-v2-fixed";const ASSETS=["/offline","/static/css/style.css","/static/js/app.js","/static/icons/icon-192.png","/static/icons/icon-512.png"];
self.addEventListener("install",e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)));self.skipWaiting()});
self.addEventListener("activate",e=>{e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))));self.clients.claim()});
self.addEventListener("fetch",e=>{if(e.request.method!=="GET")return;if(e.request.mode==="navigate"){e.respondWith(fetch(e.request).catch(()=>caches.match("/offline")));return}if(new URL(e.request.url).origin===location.origin)e.respondWith(caches.match(e.request).then(c=>c||fetch(e.request)))});
