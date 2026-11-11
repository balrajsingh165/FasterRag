---
title: fasterRag — a backend-only RAG framework for FastAPI
description: >-
  fasterRag is an open-source, backend-only Retrieval-Augmented Generation framework built on
  FastAPI. This page is the whole system on one screen: an animated map of every package, the
  path a document takes in and a question takes out, and what holds it together.
hide:
  - toc
---

# fasterRag

<div class="fr-stage">
<div class="fr-orb fr-orb--a"></div><div class="fr-orb fr-orb--b"></div><div class="fr-orb fr-orb--c"></div>
<svg class="fr-sky" viewBox="0 0 1000 520" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><line style="--i:0" x1="331" y1="88" x2="287" y2="218"/><line style="--i:1" x1="331" y1="88" x2="455" y2="161"/><line style="--i:2" x1="645" y1="50" x2="546" y2="45"/><line style="--i:3" x1="645" y1="50" x2="586" y2="143"/><line style="--i:4" x1="534" y1="194" x2="574" y2="209"/><line style="--i:5" x1="534" y1="194" x2="577" y2="238"/><line style="--i:6" x1="76" y1="264" x2="97" y2="235"/><line style="--i:7" x1="76" y1="264" x2="42" y2="241"/><line style="--i:8" x1="56" y1="227" x2="42" y2="241"/><line style="--i:0" x1="56" y1="227" x2="24" y2="220"/><line style="--i:1" x1="87" y1="59" x2="77" y2="115"/><line style="--i:2" x1="87" y1="59" x2="158" y2="72"/><line style="--i:3" x1="428" y1="421" x2="421" y2="387"/><line style="--i:4" x1="428" y1="421" x2="395" y2="443"/><line style="--i:5" x1="139" y1="124" x2="144" y2="136"/><line style="--i:6" x1="139" y1="124" x2="189" y2="128"/><line style="--i:7" x1="622" y1="480" x2="641" y2="503"/><line style="--i:8" x1="622" y1="480" x2="547" y2="449"/><line style="--i:0" x1="574" y1="209" x2="577" y2="238"/><line style="--i:1" x1="574" y1="209" x2="633" y2="197"/><line style="--i:2" x1="957" y1="37" x2="961" y2="72"/><line style="--i:3" x1="957" y1="37" x2="939" y2="88"/><line style="--i:4" x1="844" y1="156" x2="860" y2="168"/><line style="--i:5" x1="844" y1="156" x2="809" y2="154"/><line style="--i:6" x1="158" y1="72" x2="181" y2="72"/><line style="--i:7" x1="158" y1="72" x2="189" y2="128"/><line style="--i:8" x1="316" y1="416" x2="364" y2="449"/><line style="--i:0" x1="316" y1="416" x2="395" y2="443"/><line style="--i:1" x1="193" y1="300" x2="166" y2="255"/><line style="--i:2" x1="193" y1="300" x2="254" y2="297"/><line style="--i:3" x1="633" y1="197" x2="673" y2="224"/><line style="--i:4" x1="633" y1="197" x2="577" y2="238"/><line style="--i:5" x1="546" y1="45" x2="586" y2="143"/><line style="--i:6" x1="546" y1="45" x2="455" y2="161"/><line style="--i:7" x1="77" y1="115" x2="144" y2="136"/><line style="--i:8" x1="77" y1="115" x2="189" y2="128"/><line style="--i:0" x1="673" y1="224" x2="687" y2="306"/><line style="--i:1" x1="673" y1="224" x2="720" y2="156"/><line style="--i:2" x1="322" y1="302" x2="254" y2="297"/><line style="--i:3" x1="322" y1="302" x2="390" y2="343"/><line style="--i:4" x1="455" y1="161" x2="586" y2="143"/><line style="--i:5" x1="455" y1="161" x2="577" y2="238"/><line style="--i:6" x1="783" y1="358" x2="754" y2="296"/><line style="--i:7" x1="783" y1="358" x2="807" y2="439"/><line style="--i:8" x1="254" y1="297" x2="244" y2="253"/><line style="--i:0" x1="254" y1="297" x2="287" y2="218"/><line style="--i:1" x1="524" y1="445" x2="547" y2="449"/><line style="--i:2" x1="524" y1="445" x2="475" y2="341"/><line style="--i:3" x1="720" y1="156" x2="809" y2="154"/><line style="--i:4" x1="720" y1="156" x2="586" y2="143"/><line style="--i:5" x1="961" y1="72" x2="939" y2="88"/><line style="--i:6" x1="961" y1="72" x2="860" y2="168"/><line style="--i:7" x1="421" y1="387" x2="390" y2="343"/><line style="--i:8" x1="421" y1="387" x2="395" y2="443"/><line style="--i:0" x1="166" y1="255" x2="97" y2="235"/><line style="--i:1" x1="166" y1="255" x2="244" y2="253"/><line style="--i:2" x1="58" y1="343" x2="78" y2="359"/><line style="--i:3" x1="58" y1="343" x2="77" y2="392"/><line style="--i:4" x1="754" y1="296" x2="687" y2="306"/><line style="--i:5" x1="860" y1="168" x2="809" y2="154"/><line style="--i:6" x1="860" y1="168" x2="939" y2="88"/><line style="--i:7" x1="687" y1="306" x2="577" y2="238"/><line style="--i:8" x1="577" y1="238" x2="586" y2="143"/><line style="--i:0" x1="577" y1="238" x2="475" y2="341"/><line style="--i:1" x1="826" y1="479" x2="807" y2="439"/><line style="--i:2" x1="475" y1="341" x2="390" y2="343"/><line style="--i:3" x1="475" y1="341" x2="395" y2="443"/><line style="--i:4" x1="78" y1="359" x2="77" y2="392"/><line style="--i:5" x1="78" y1="359" x2="42" y2="241"/><line style="--i:6" x1="641" y1="503" x2="547" y2="449"/><line style="--i:7" x1="809" y1="154" x2="939" y2="88"/><line style="--i:8" x1="390" y1="343" x2="395" y2="443"/><line style="--i:0" x1="390" y1="343" x2="364" y2="449"/><line style="--i:1" x1="42" y1="241" x2="24" y2="220"/><line style="--i:2" x1="42" y1="241" x2="97" y2="235"/><line style="--i:3" x1="181" y1="72" x2="189" y2="128"/><line style="--i:4" x1="181" y1="72" x2="144" y2="136"/><line style="--i:5" x1="144" y1="136" x2="189" y2="128"/><line style="--i:6" x1="144" y1="136" x2="97" y2="235"/><line style="--i:7" x1="395" y1="443" x2="364" y2="449"/><line style="--i:8" x1="97" y1="235" x2="24" y2="220"/><line style="--i:0" x1="97" y1="235" x2="189" y2="128"/><line style="--i:1" x1="287" y1="218" x2="244" y2="253"/><line style="--i:2" x1="287" y1="218" x2="189" y2="128"/><line style="--i:3" x1="189" y1="128" x2="244" y2="253"/><circle style="--i:0" cx="331" cy="88" r="1.4"/><circle style="--i:1" cx="645" cy="50" r="1.9"/><circle style="--i:2" cx="534" cy="194" r="2.4"/><circle style="--i:3" cx="76" cy="264" r="2.9"/><circle style="--i:4" cx="56" cy="227" r="1.4"/><circle style="--i:5" cx="87" cy="59" r="1.9"/><circle style="--i:6" cx="428" cy="421" r="2.4"/><circle style="--i:7" cx="139" cy="124" r="2.9"/><circle style="--i:8" cx="622" cy="480" r="1.4"/><circle style="--i:9" cx="574" cy="209" r="1.9"/><circle style="--i:10" cx="957" cy="37" r="2.4"/><circle style="--i:0" cx="844" cy="156" r="2.9"/><circle style="--i:1" cx="158" cy="72" r="1.4"/><circle style="--i:2" cx="316" cy="416" r="1.9"/><circle style="--i:3" cx="193" cy="300" r="2.4"/><circle style="--i:4" cx="633" cy="197" r="2.9"/><circle style="--i:5" cx="546" cy="45" r="1.4"/><circle style="--i:6" cx="77" cy="115" r="1.9"/><circle style="--i:7" cx="673" cy="224" r="2.4"/><circle style="--i:8" cx="322" cy="302" r="2.9"/><circle style="--i:9" cx="455" cy="161" r="1.4"/><circle style="--i:10" cx="783" cy="358" r="1.9"/><circle style="--i:0" cx="254" cy="297" r="2.4"/><circle style="--i:1" cx="524" cy="445" r="2.9"/><circle style="--i:2" cx="720" cy="156" r="1.4"/><circle style="--i:3" cx="961" cy="72" r="1.9"/><circle style="--i:4" cx="421" cy="387" r="2.4"/><circle style="--i:5" cx="166" cy="255" r="2.9"/><circle style="--i:6" cx="58" cy="343" r="1.4"/><circle style="--i:7" cx="754" cy="296" r="1.9"/><circle style="--i:8" cx="860" cy="168" r="2.4"/><circle style="--i:9" cx="687" cy="306" r="2.9"/><circle style="--i:10" cx="577" cy="238" r="1.4"/><circle style="--i:0" cx="826" cy="479" r="1.9"/><circle style="--i:1" cx="475" cy="341" r="2.4"/><circle style="--i:2" cx="78" cy="359" r="2.9"/><circle style="--i:3" cx="641" cy="503" r="1.4"/><circle style="--i:4" cx="809" cy="154" r="1.9"/><circle style="--i:5" cx="390" cy="343" r="2.4"/><circle style="--i:6" cx="42" cy="241" r="2.9"/><circle style="--i:7" cx="181" cy="72" r="1.4"/><circle style="--i:8" cx="77" cy="392" r="1.9"/><circle style="--i:9" cx="144" cy="136" r="2.4"/><circle style="--i:10" cx="395" cy="443" r="2.9"/><circle style="--i:0" cx="97" cy="235" r="1.4"/><circle style="--i:1" cx="547" cy="449" r="1.9"/><circle style="--i:2" cx="807" cy="439" r="2.4"/><circle style="--i:3" cx="287" cy="218" r="2.9"/><circle style="--i:4" cx="364" cy="449" r="1.4"/><circle style="--i:5" cx="939" cy="88" r="1.9"/><circle style="--i:6" cx="189" cy="128" r="2.4"/><circle style="--i:7" cx="244" cy="253" r="2.9"/><circle style="--i:8" cx="586" cy="143" r="1.4"/><circle style="--i:9" cx="24" cy="220" r="1.9"/></svg>
<div class="fr-beam"></div>
<div class="fr-hero">
<div class="fr-mark">
<svg class="fr-rings" viewBox="0 0 220 220" aria-hidden="true"><circle cx="110" cy="110" r="100"/><circle cx="110" cy="110" r="82"/><circle cx="110" cy="110" r="64"/></svg>
<svg class="fr-logo" viewBox="0 0 128 128" role="img" aria-label="The fasterRag logo: three chunk bars streaming into a lightning bolt"><defs><linearGradient id="fr-tile" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#38bdf8"/></linearGradient></defs><rect width="128" height="128" rx="28" fill="url(#fr-tile)"/><g fill="#fff"><rect class="fr-bar" style="--i:0" x="18" y="40" width="34" height="9" rx="4.5"/><rect class="fr-bar" style="--i:1" x="12" y="59.5" width="30" height="9" rx="4.5"/><rect class="fr-bar" style="--i:2" x="20" y="79" width="22" height="9" rx="4.5"/><path class="fr-bolt" d="M84 14 52 68h20l-8 46 38-58H80z"/></g></svg>
</div>
<p class="fr-title">faster<span>Rag</span></p>
<p class="fr-lede">A backend-only RAG system you drive from a REST API, a terminal or Python. Documents go in through a pipeline that records a document as indexed only when it is; questions come out with citations and a trace you can replay.</p>
<div class="fr-term" aria-hidden="true">
<div class="fr-term-bar"><i></i><i></i><i></i><span>the whole lifecycle, from a terminal</span></div>
<div class="fr-term-body"><span class="fr-l fr-l1">$ fasterrag config init</span><span class="fr-l fr-l2">$ fasterrag doctor</span><span class="fr-l fr-l3">$ fasterrag ingest ./handbook --recursive --watch</span><span class="fr-l fr-l4">$ fasterrag query "What is the refund window?" --show-chunks</span></div>
</div>
<div class="fr-facts">
<span><b>3</b> ways in: REST, CLI, Python</span>
<span><b>5</b> vector databases</span>
<span><b>4</b> embedding providers</span>
<span><b>4</b> LLM SDKs and any OpenAI-compatible URL</span>
<span><b>7</b> document formats</span>
<span><b>5</b> chunking strategies</span>
</div>
</div>
</div>

[Run the quickstart](quickstart.md){ .md-button .md-button--primary }
[Architecture in depth](architecture.md){ .md-button }
[What is and is not built](release-readiness.md){ .md-button }
[View on GitHub](https://github.com/balrajsingh165/fasterrag){ .md-button }

<div class="fr-jump" markdown>
[The map](#the-whole-system-on-one-map) · [A document goes in](#a-document-goes-in) · [A question comes out](#a-question-comes-out) · [What holds it together](#what-holds-it-together) · [Every module](#every-package-module-by-module) · [Configuration](#one-file-decides-the-vendors) · [Honest status](#honest-status)
</div>

## The whole system on one map

Dependencies point **down**. A surface calls services; services drive the engine; the engine and the
services reach the outside world only through an adapter. Bars moving through the map are an
ingestion; dots are a query. Pick one to follow, or hover a box to isolate it.

<div class="fr-map">
<div class="fr-modes" role="radiogroup" aria-label="Which flow to show on the map">
<input type="radio" name="fr-mode" id="fr-m-both" checked><label for="fr-m-both">Both flows</label>
<input type="radio" name="fr-mode" id="fr-m-in"><label for="fr-m-in"><i class="fr-key--in"></i>Ingestion</label>
<input type="radio" name="fr-mode" id="fr-m-q"><label for="fr-m-q"><i class="fr-key--q"></i>Query</label>
</div>
<div class="fr-map-scroll">
<svg viewBox="0 0 1000 2054" role="img" aria-labelledby="fr-map-title fr-map-desc" xmlns="http://www.w3.org/2000/svg"><title id="fr-map-title">Map of the fasterRag repository</title><desc id="fr-map-desc">Three entry surfaces — REST API, CLI and Python package — call one service layer. Services drive the worker pipeline and the core library and reach vector databases, embedding models and LLM providers only through adapters. Configuration, the error taxonomy and observability are used by every layer. Tests, documentation, gate scripts and deployment files sit around the package. Moving bars trace an ingestion; moving dots trace a query.</desc><defs><linearGradient id="fr-fill" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#0ea5e9"/></linearGradient></defs><path class="fr-wire" d="M148.0 106.0 V178.0"/><path class="fr-wire" d="M400.0 106.0 V178.0"/><path class="fr-wire" d="M652.0 106.0 V178.0"/><path class="fr-wire fr-wire--in" d="M148.0 566.0 V638.0"/><path class="fr-wire" d="M400.0 566.0 V638.0"/><path class="fr-wire fr-wire--q" d="M652.0 566.0 V638.0"/><path class="fr-wire fr-wire--in" d="M205.0 884.0 V956.0"/><path class="fr-wire" d="M587.0 884.0 V956.0"/><path class="fr-wire" d="M380 986.0 H404"/><path class="fr-wire fr-wire--in" d="M205.0 1391.0 V1463.0"/><path class="fr-wire fr-wire--in" d="M350 1391.0 C350 1431.0 470 1423.0 470 1463.0"/><path class="fr-wire fr-wire--q" d="M392.0 884.0 V1463.0"/><path class="fr-wire fr-wire--q" d="M392.0 1395.0 C392.0 1435.0 290 1423.0 290 1463.0"/><path class="fr-wire fr-wire--q" d="M780.0 884.0 V1463.0"/><path class="fr-wire" d="M180.0 1655.0 V1727.0"/><path class="fr-wire" d="M500.0 1655.0 V1727.0"/><path class="fr-wire" d="M820.0 1655.0 V1727.0"/><path class="fr-wire fr-wire--rail" d="M770 212.0 H790"/><path class="fr-wire fr-wire--rail" d="M770 672.0 H790"/><path class="fr-wire fr-wire--rail" d="M770 990.0 H790"/><text class="fr-band" x="30" y="26">WHO CALLS IT</text><text class="fr-band" x="30" y="164">THREE SURFACES — THIN ON PURPOSE</text><text class="fr-band" x="30" y="624">ONE SERVICE LAYER — THE SAME BEHAVIOUR ON EVERY SURFACE</text><text class="fr-band" x="30" y="942">THE ENGINE</text><text class="fr-band" x="30" y="1449">THE VENDOR BOUNDARY — NO VENDOR TYPE LEAKS PAST IT</text><text class="fr-band" x="30" y="1713">OUTSIDE THE PROCESS</text><text class="fr-band" x="30" y="1857">AROUND THE PACKAGE</text><text class="fr-band" x="790" y="164">USED BY EVERY LAYER</text><g class="fr-node fr-node--caller fr-float" style="--i:0"><rect x="30" y="40" width="236" height="66" rx="14"/><text class="fr-name" x="46" y="69.0">HTTP client</text><text class="fr-path" x="46" y="86.0">curl · your service</text></g><g class="fr-node fr-node--caller fr-float" style="--i:1"><rect x="282" y="40" width="236" height="66" rx="14"/><text class="fr-name" x="298" y="69.0">Terminal</text><text class="fr-path" x="502" y="68.0" text-anchor="end">operator · CI job</text></g><g class="fr-node fr-node--caller fr-float" style="--i:2"><rect x="534" y="40" width="236" height="66" rx="14"/><text class="fr-name" x="550" y="69.0">Python code</text><text class="fr-path" x="754" y="68.0" text-anchor="end">script · app</text></g><g class="fr-node fr-node--surface fr-float" style="--i:3"><rect x="30" y="178" width="236" height="388" rx="14"/><text class="fr-name" x="46" y="207.0">REST API</text><text class="fr-path" x="250" y="206.0" text-anchor="end">api/</text><text class="fr-group" x="46" y="232.0">ROUTES</text><g class="fr-chip"><rect x="44" y="239" width="83" height="22" rx="11"/><text x="85.5" y="254" text-anchor="middle">/v1/ingest</text></g><g class="fr-chip"><rect x="133" y="239" width="76" height="22" rx="11"/><text x="171.0" y="254" text-anchor="middle">/v1/query</text></g><g class="fr-chip"><rect x="44" y="267" width="96" height="22" rx="11"/><text x="92.0" y="282" text-anchor="middle">/v1/retrieve</text></g><g class="fr-chip"><rect x="44" y="295" width="116" height="22" rx="11"/><text x="102.0" y="310" text-anchor="middle">/v1/collections</text></g><g class="fr-chip"><rect x="166" y="295" width="83" height="22" rx="11"/><text x="207.5" y="310" text-anchor="middle">/v1/traces</text></g><g class="fr-chip"><rect x="44" y="323" width="83" height="22" rx="11"/><text x="85.5" y="338" text-anchor="middle">/v1/replay</text></g><g class="fr-chip"><rect x="133" y="323" width="96" height="22" rx="11"/><text x="181.0" y="338" text-anchor="middle">/v1/estimate</text></g><g class="fr-chip"><rect x="44" y="351" width="90" height="22" rx="11"/><text x="89.0" y="366" text-anchor="middle">/v1/admin/*</text></g><g class="fr-chip"><rect x="140" y="351" width="70" height="22" rx="11"/><text x="175.0" y="366" text-anchor="middle">/healthz</text></g><g class="fr-chip"><rect x="44" y="379" width="63" height="22" rx="11"/><text x="75.5" y="394" text-anchor="middle">/readyz</text></g><g class="fr-chip"><rect x="113" y="379" width="70" height="22" rx="11"/><text x="148.0" y="394" text-anchor="middle">/metrics</text></g><text class="fr-group" x="46" y="421.0">IN FRONT OF EVERY ROUTE</text><g class="fr-chip"><rect x="44" y="428" width="103" height="22" rx="11"/><text x="95.5" y="443" text-anchor="middle">auth + scopes</text></g><g class="fr-chip"><rect x="153" y="428" width="83" height="22" rx="11"/><text x="194.5" y="443" text-anchor="middle">rate limit</text></g><g class="fr-chip"><rect x="44" y="456" width="83" height="22" rx="11"/><text x="85.5" y="471" text-anchor="middle">body limit</text></g><g class="fr-chip"><rect x="133" y="456" width="90" height="22" rx="11"/><text x="178.0" y="471" text-anchor="middle">idempotency</text></g><g class="fr-chip"><rect x="44" y="484" width="110" height="22" rx="11"/><text x="99.0" y="499" text-anchor="middle">correlation id</text></g></g><g class="fr-node fr-node--surface fr-float" style="--i:4"><rect x="282" y="178" width="236" height="388" rx="14"/><text class="fr-name" x="298" y="207.0">CLI</text><text class="fr-path" x="502" y="206.0" text-anchor="end">cli/</text><text class="fr-group" x="298" y="232.0">MOVE DATA</text><g class="fr-chip"><rect x="296" y="239" width="56" height="22" rx="11"/><text x="324.0" y="254" text-anchor="middle">ingest</text></g><g class="fr-chip"><rect x="358" y="239" width="50" height="22" rx="11"/><text x="383.0" y="254" text-anchor="middle">query</text></g><g class="fr-chip"><rect x="414" y="239" width="50" height="22" rx="11"/><text x="439.0" y="254" text-anchor="middle">index</text></g><g class="fr-chip"><rect x="296" y="267" width="56" height="22" rx="11"/><text x="324.0" y="282" text-anchor="middle">export</text></g><g class="fr-chip"><rect x="358" y="267" width="56" height="22" rx="11"/><text x="386.0" y="282" text-anchor="middle">import</text></g><text class="fr-group" x="298" y="309.0">INSPECT</text><g class="fr-chip"><rect x="296" y="316" width="56" height="22" rx="11"/><text x="324.0" y="331" text-anchor="middle">doctor</text></g><g class="fr-chip"><rect x="358" y="316" width="56" height="22" rx="11"/><text x="386.0" y="331" text-anchor="middle">status</text></g><g class="fr-chip"><rect x="420" y="316" width="56" height="22" rx="11"/><text x="448.0" y="331" text-anchor="middle">traces</text></g><g class="fr-chip"><rect x="296" y="344" width="56" height="22" rx="11"/><text x="324.0" y="359" text-anchor="middle">replay</text></g><g class="fr-chip"><rect x="358" y="344" width="70" height="22" rx="11"/><text x="393.0" y="359" text-anchor="middle">estimate</text></g><g class="fr-chip"><rect x="434" y="344" width="56" height="22" rx="11"/><text x="462.0" y="359" text-anchor="middle">config</text></g><text class="fr-group" x="298" y="386.0">OPERATE</text><g class="fr-chip"><rect x="296" y="393" width="50" height="22" rx="11"/><text x="321.0" y="408" text-anchor="middle">serve</text></g><g class="fr-chip"><rect x="352" y="393" width="76" height="22" rx="11"/><text x="390.0" y="408" text-anchor="middle">provision</text></g><g class="fr-chip"><rect x="434" y="393" width="56" height="22" rx="11"/><text x="462.0" y="408" text-anchor="middle">backup</text></g><g class="fr-chip"><rect x="296" y="421" width="63" height="22" rx="11"/><text x="327.5" y="436" text-anchor="middle">restore</text></g><g class="fr-chip"><rect x="365" y="421" width="76" height="22" rx="11"/><text x="403.0" y="436" text-anchor="middle">benchmark</text></g><g class="fr-chip"><rect x="296" y="449" width="76" height="22" rx="11"/><text x="334.0" y="464" text-anchor="middle">autopilot</text></g></g><g class="fr-node fr-node--surface fr-float" style="--i:5"><rect x="534" y="178" width="236" height="388" rx="14"/><text class="fr-name" x="550" y="207.0">Python package</text><text class="fr-path" x="550" y="224.0">facade.py · sync.py</text><text class="fr-group" x="550" y="248.0">FASTERRAG</text><g class="fr-chip"><rect x="548" y="255" width="76" height="22" rx="11"/><text x="586.0" y="270" text-anchor="middle">.ingest()</text></g><g class="fr-chip"><rect x="630" y="255" width="70" height="22" rx="11"/><text x="665.0" y="270" text-anchor="middle">.query()</text></g><g class="fr-chip"><rect x="548" y="283" width="116" height="22" rx="11"/><text x="606.0" y="298" text-anchor="middle">.query_stream()</text></g><g class="fr-chip"><rect x="548" y="311" width="90" height="22" rx="11"/><text x="593.0" y="326" text-anchor="middle">.retrieve()</text></g><g class="fr-chip"><rect x="644" y="311" width="76" height="22" rx="11"/><text x="682.0" y="326" text-anchor="middle">.replay()</text></g><g class="fr-chip"><rect x="548" y="339" width="90" height="22" rx="11"/><text x="593.0" y="354" text-anchor="middle">.estimate()</text></g><g class="fr-chip"><rect x="644" y="339" width="76" height="22" rx="11"/><text x="682.0" y="354" text-anchor="middle">.doctor()</text></g><g class="fr-chip"><rect x="548" y="367" width="110" height="22" rx="11"/><text x="603.0" y="382" text-anchor="middle">.collections()</text></g><text class="fr-group" x="550" y="409.0">STANDALONE PARTS</text><g class="fr-chip"><rect x="548" y="416" width="137" height="22" rx="11"/><text x="616.5" y="431" text-anchor="middle">fasterrag.chunking</text></g><g class="fr-chip"><rect x="548" y="444" width="130" height="22" rx="11"/><text x="613.0" y="459" text-anchor="middle">fasterrag.parsing</text></g><g class="fr-chip"><rect x="548" y="472" width="143" height="22" rx="11"/><text x="619.5" y="487" text-anchor="middle">fasterrag.retrieval</text></g><g class="fr-chip"><rect x="548" y="500" width="123" height="22" rx="11"/><text x="609.5" y="515" text-anchor="middle">fasterrag.rerank</text></g><g class="fr-chip"><rect x="548" y="528" width="116" height="22" rx="11"/><text x="606.0" y="543" text-anchor="middle">fasterrag.evals</text></g></g><g class="fr-node fr-node--rail fr-float" style="--i:6"><rect x="790" y="178" width="180" height="232" rx="14"/><text class="fr-name" x="806" y="207.0">Config</text><text class="fr-path" x="954" y="206.0" text-anchor="end">config/</text><text class="fr-group" x="806" y="232.0">VALIDATED AT LOAD</text><g class="fr-chip"><rect x="804" y="239" width="76" height="22" rx="11"/><text x="842.0" y="254" text-anchor="middle">schema.py</text></g><g class="fr-chip"><rect x="804" y="267" width="76" height="22" rx="11"/><text x="842.0" y="282" text-anchor="middle">loader.py</text></g><g class="fr-chip"><rect x="804" y="295" width="90" height="22" rx="11"/><text x="849.0" y="310" text-anchor="middle">template.py</text></g><text class="fr-group" x="806" y="337.0">TWO FILES</text><g class="fr-chip"><rect x="804" y="344" width="90" height="22" rx="11"/><text x="849.0" y="359" text-anchor="middle">config.yaml</text></g><g class="fr-chip"><rect x="804" y="372" width="96" height="22" rx="11"/><text x="852.0" y="387" text-anchor="middle">.env secrets</text></g></g><g class="fr-node fr-float" style="--i:7"><rect x="30" y="638" width="740" height="246" rx="14"/><text class="fr-name" x="46" y="667.0">Services</text><text class="fr-path" x="754" y="666.0" text-anchor="end">services/</text><text class="fr-group" x="46" y="692.0">INGEST AND STATE</text><g class="fr-chip"><rect x="44" y="699" width="76" height="22" rx="11"/><text x="82.0" y="714" text-anchor="middle">ingestion</text></g><g class="fr-chip"><rect x="126" y="699" width="63" height="22" rx="11"/><text x="157.5" y="714" text-anchor="middle">sources</text></g><g class="fr-chip"><rect x="195" y="699" width="63" height="22" rx="11"/><text x="226.5" y="714" text-anchor="middle">journal</text></g><g class="fr-chip"><rect x="264" y="699" width="90" height="22" rx="11"/><text x="309.0" y="714" text-anchor="middle">idempotency</text></g><g class="fr-chip"><rect x="360" y="699" width="70" height="22" rx="11"/><text x="395.0" y="714" text-anchor="middle">lockfile</text></g><g class="fr-chip"><rect x="436" y="699" width="63" height="22" rx="11"/><text x="467.5" y="714" text-anchor="middle">tenancy</text></g><g class="fr-chip"><rect x="505" y="699" width="70" height="22" rx="11"/><text x="540.0" y="714" text-anchor="middle">backfill</text></g><text class="fr-group" x="46" y="741.0">QUERY</text><g class="fr-chip"><rect x="44" y="748" width="70" height="22" rx="11"/><text x="79.0" y="763" text-anchor="middle">querying</text></g><g class="fr-chip"><rect x="120" y="748" width="83" height="22" rx="11"/><text x="161.5" y="763" text-anchor="middle">generation</text></g><g class="fr-chip"><rect x="209" y="748" width="90" height="22" rx="11"/><text x="254.0" y="763" text-anchor="middle">degradation</text></g><g class="fr-chip"><rect x="305" y="748" width="56" height="22" rx="11"/><text x="333.0" y="763" text-anchor="middle">budget</text></g><g class="fr-chip"><rect x="367" y="748" width="56" height="22" rx="11"/><text x="395.0" y="763" text-anchor="middle">traces</text></g><g class="fr-chip"><rect x="429" y="748" width="56" height="22" rx="11"/><text x="457.0" y="763" text-anchor="middle">replay</text></g><text class="fr-group" x="46" y="790.0">QUALITY</text><g class="fr-chip"><rect x="44" y="797" width="83" height="22" rx="11"/><text x="85.5" y="812" text-anchor="middle">evaluation</text></g><g class="fr-chip"><rect x="133" y="797" width="83" height="22" rx="11"/><text x="174.5" y="812" text-anchor="middle">regression</text></g><g class="fr-chip"><rect x="222" y="797" width="90" height="22" rx="11"/><text x="267.0" y="812" text-anchor="middle">golden_sets</text></g><g class="fr-chip"><rect x="318" y="797" width="76" height="22" rx="11"/><text x="356.0" y="812" text-anchor="middle">autopilot</text></g><g class="fr-chip"><rect x="400" y="797" width="76" height="22" rx="11"/><text x="438.0" y="812" text-anchor="middle">benchmark</text></g><g class="fr-chip"><rect x="482" y="797" width="83" height="22" rx="11"/><text x="523.5" y="812" text-anchor="middle">estimation</text></g><text class="fr-group" x="46" y="839.0">OPERATIONS</text><g class="fr-chip"><rect x="44" y="846" width="56" height="22" rx="11"/><text x="72.0" y="861" text-anchor="middle">doctor</text></g><g class="fr-chip"><rect x="106" y="846" width="63" height="22" rx="11"/><text x="137.5" y="861" text-anchor="middle">reindex</text></g><g class="fr-chip"><rect x="175" y="846" width="56" height="22" rx="11"/><text x="203.0" y="861" text-anchor="middle">backup</text></g><g class="fr-chip"><rect x="237" y="846" width="63" height="22" rx="11"/><text x="268.5" y="861" text-anchor="middle">archive</text></g><g class="fr-chip"><rect x="306" y="846" width="110" height="22" rx="11"/><text x="361.0" y="861" text-anchor="middle">archive_import</text></g><g class="fr-chip"><rect x="422" y="846" width="96" height="22" rx="11"/><text x="470.0" y="861" text-anchor="middle">provisioning</text></g></g><g class="fr-node fr-node--rail fr-float" style="--i:8"><rect x="790" y="638" width="180" height="183" rx="14"/><text class="fr-name" x="806" y="667.0">Errors</text><text class="fr-path" x="954" y="666.0" text-anchor="end">errors.py</text><text class="fr-group" x="806" y="692.0">ONE TAXONOMY</text><g class="fr-chip"><rect x="804" y="699" width="90" height="22" rx="11"/><text x="849.0" y="714" text-anchor="middle">typed codes</text></g><g class="fr-chip"><rect x="804" y="727" width="110" height="22" rx="11"/><text x="859.0" y="742" text-anchor="middle">retryable flag</text></g><g class="fr-chip"><rect x="804" y="755" width="96" height="22" rx="11"/><text x="852.0" y="770" text-anchor="middle">problem+json</text></g><g class="fr-chip"><rect x="804" y="783" width="70" height="22" rx="11"/><text x="839.0" y="798" text-anchor="middle">trace id</text></g></g><g class="fr-node fr-float" style="--i:9"><rect x="30" y="956" width="350" height="435" rx="14"/><text class="fr-name" x="46" y="985.0">Workers</text><text class="fr-path" x="364" y="984.0" text-anchor="end">workers/</text><rect class="fr-stagebox" x="44.0" y="1000.0" width="71.5" height="50" rx="9"/><text class="fr-stage-name" x="79.8" y="1019.0" text-anchor="middle">cpu_pool</text><text class="fr-stage-sub" x="79.8" y="1037.0" text-anchor="middle">parse · chunk</text><path class="fr-wire fr-wire--in" d="M116.5 1025.0 H126.5"/><rect class="fr-stagebox" x="127.5" y="1000.0" width="71.5" height="50" rx="9"/><text class="fr-stage-name" x="163.2" y="1019.0" text-anchor="middle">queues</text><rect class="fr-slot" style="--i:0" x="136.2" y="1028.0" width="8" height="13" rx="2"/><rect class="fr-slot" style="--i:1" x="147.2" y="1028.0" width="8" height="13" rx="2"/><rect class="fr-slot" style="--i:2" x="158.2" y="1028.0" width="8" height="13" rx="2"/><rect class="fr-slot" style="--i:3" x="169.2" y="1028.0" width="8" height="13" rx="2"/><rect class="fr-slot" style="--i:4" x="180.2" y="1028.0" width="8" height="13" rx="2"/><path class="fr-wire fr-wire--in" d="M200.0 1025.0 H210.0"/><rect class="fr-stagebox" x="211.0" y="1000.0" width="71.5" height="50" rx="9"/><text class="fr-stage-name" x="246.8" y="1019.0" text-anchor="middle">embed_pool</text><text class="fr-stage-sub" x="246.8" y="1037.0" text-anchor="middle">batch · retry</text><path class="fr-wire fr-wire--in" d="M283.5 1025.0 H293.5"/><rect class="fr-stagebox" x="294.5" y="1000.0" width="71.5" height="50" rx="9"/><text class="fr-stage-name" x="330.2" y="1019.0" text-anchor="middle">indexer</text><text class="fr-stage-sub" x="330.2" y="1037.0" text-anchor="middle">upsert</text><rect class="fr-ledger" x="44" y="1060.0" width="322" height="30" rx="9"/><text class="fr-stage-name" x="56" y="1079.0">ledger</text><text class="fr-stage-sub" x="106" y="1079.0">indexed only when the last chunk is written</text><circle class="fr-tick" style="--i:0" cx="298" cy="1075.0" r="4"/><circle class="fr-tick" style="--i:1" cx="311" cy="1075.0" r="4"/><circle class="fr-tick" style="--i:2" cx="324" cy="1075.0" r="4"/><circle class="fr-tick" style="--i:3" cx="337" cy="1075.0" r="4"/><circle class="fr-tick" style="--i:4" cx="350" cy="1075.0" r="4"/><text class="fr-group" x="46" y="1108.0">MODULES</text><g class="fr-chip"><rect x="44" y="1115" width="90" height="22" rx="11"/><text x="89.0" y="1130" text-anchor="middle">cpu_pool.py</text></g><g class="fr-chip"><rect x="140" y="1115" width="76" height="22" rx="11"/><text x="178.0" y="1130" text-anchor="middle">queues.py</text></g><g class="fr-chip"><rect x="222" y="1115" width="103" height="22" rx="11"/><text x="273.5" y="1130" text-anchor="middle">embed_pool.py</text></g><g class="fr-chip"><rect x="44" y="1143" width="83" height="22" rx="11"/><text x="85.5" y="1158" text-anchor="middle">indexer.py</text></g><g class="fr-chip"><rect x="133" y="1143" width="76" height="22" rx="11"/><text x="171.0" y="1158" text-anchor="middle">ledger.py</text></g><text class="fr-group" x="46" y="1185.0">WHAT THE PIPELINE PROMISES</text><g class="fr-chip"><rect x="44" y="1192" width="110" height="22" rx="11"/><text x="99.0" y="1207" text-anchor="middle">bounded memory</text></g><g class="fr-chip"><rect x="44" y="1220" width="210" height="22" rx="11"/><text x="149.0" y="1235" text-anchor="middle">one bad file, one dead letter</text></g><g class="fr-chip"><rect x="44" y="1248" width="137" height="22" rx="11"/><text x="112.5" y="1263" text-anchor="middle">retry with backoff</text></g><g class="fr-chip"><rect x="187" y="1248" width="170" height="22" rx="11"/><text x="272.0" y="1263" text-anchor="middle">deterministic chunk ids</text></g><g class="fr-chip"><rect x="44" y="1276" width="150" height="22" rx="11"/><text x="119.0" y="1291" text-anchor="middle">a job always settles</text></g><text class="fr-group" x="46" y="1318.0">RUNS IN</text><g class="fr-chip"><rect x="44" y="1325" width="190" height="22" rx="11"/><text x="139.0" y="1340" text-anchor="middle">a process pool for parsing</text></g><g class="fr-chip"><rect x="44" y="1353" width="184" height="22" rx="11"/><text x="136.0" y="1368" text-anchor="middle">async tasks for embedding</text></g></g><g class="fr-node fr-float" style="--i:10"><rect x="404" y="956" width="366" height="435" rx="14"/><text class="fr-name" x="420" y="985.0">Core</text><text class="fr-path" x="754" y="984.0" text-anchor="end">core/</text><text class="fr-group" x="420" y="1010.0">PARSING</text><g class="fr-chip"><rect x="418" y="1017" width="36" height="22" rx="11"/><text x="436.0" y="1032" text-anchor="middle">pdf</text></g><g class="fr-chip"><rect x="460" y="1017" width="43" height="22" rx="11"/><text x="481.5" y="1032" text-anchor="middle">html</text></g><g class="fr-chip"><rect x="509" y="1017" width="70" height="22" rx="11"/><text x="544.0" y="1032" text-anchor="middle">markdown</text></g><g class="fr-chip"><rect x="585" y="1017" width="43" height="22" rx="11"/><text x="606.5" y="1032" text-anchor="middle">docx</text></g><g class="fr-chip"><rect x="634" y="1017" width="76" height="22" rx="11"/><text x="672.0" y="1032" text-anchor="middle">plaintext</text></g><g class="fr-chip"><rect x="418" y="1045" width="83" height="22" rx="11"/><text x="459.5" y="1060" text-anchor="middle">csv · json</text></g><text class="fr-group" x="420" y="1087.0">CHUNKING</text><g class="fr-chip"><rect x="418" y="1094" width="50" height="22" rx="11"/><text x="443.0" y="1109" text-anchor="middle">fixed</text></g><g class="fr-chip"><rect x="474" y="1094" width="76" height="22" rx="11"/><text x="512.0" y="1109" text-anchor="middle">recursive</text></g><g class="fr-chip"><rect x="556" y="1094" width="70" height="22" rx="11"/><text x="591.0" y="1109" text-anchor="middle">semantic</text></g><g class="fr-chip"><rect x="632" y="1094" width="56" height="22" rx="11"/><text x="660.0" y="1109" text-anchor="middle">layout</text></g><g class="fr-chip"><rect x="694" y="1094" width="43" height="22" rx="11"/><text x="715.5" y="1109" text-anchor="middle">late</text></g><g class="fr-chip"><rect x="418" y="1122" width="83" height="22" rx="11"/><text x="459.5" y="1137" text-anchor="middle">enrichment</text></g><text class="fr-group" x="420" y="1164.0">RETRIEVAL</text><g class="fr-chip"><rect x="418" y="1171" width="43" height="22" rx="11"/><text x="439.5" y="1186" text-anchor="middle">bm25</text></g><g class="fr-chip"><rect x="467" y="1171" width="56" height="22" rx="11"/><text x="495.0" y="1186" text-anchor="middle">fusion</text></g><g class="fr-chip"><rect x="529" y="1171" width="56" height="22" rx="11"/><text x="557.0" y="1186" text-anchor="middle">rerank</text></g><g class="fr-chip"><rect x="591" y="1171" width="63" height="22" rx="11"/><text x="622.5" y="1186" text-anchor="middle">context</text></g><text class="fr-group" x="420" y="1213.0">ANSWERING</text><g class="fr-chip"><rect x="418" y="1220" width="83" height="22" rx="11"/><text x="459.5" y="1235" text-anchor="middle">generation</text></g><g class="fr-chip"><rect x="507" y="1220" width="96" height="22" rx="11"/><text x="555.0" y="1235" text-anchor="middle">faithfulness</text></g><g class="fr-chip"><rect x="609" y="1220" width="63" height="22" rx="11"/><text x="640.5" y="1235" text-anchor="middle">tracing</text></g><text class="fr-group" x="420" y="1262.0">GUARDS</text><g class="fr-chip"><rect x="418" y="1269" width="63" height="22" rx="11"/><text x="449.5" y="1284" text-anchor="middle">breaker</text></g><g class="fr-chip"><rect x="487" y="1269" width="83" height="22" rx="11"/><text x="528.5" y="1284" text-anchor="middle">rate_limit</text></g><g class="fr-chip"><rect x="576" y="1269" width="50" height="22" rx="11"/><text x="601.0" y="1284" text-anchor="middle">cache</text></g><g class="fr-chip"><rect x="632" y="1269" width="70" height="22" rx="11"/><text x="667.0" y="1284" text-anchor="middle">identity</text></g><g class="fr-chip"><rect x="418" y="1297" width="50" height="22" rx="11"/><text x="443.0" y="1312" text-anchor="middle">evals</text></g></g><g class="fr-node fr-node--rail fr-float" style="--i:11"><rect x="790" y="956" width="180" height="199" rx="14"/><text class="fr-name" x="806" y="985.0">Observability</text><text class="fr-path" x="806" y="1002.0">observability/</text><text class="fr-group" x="806" y="1026.0">SIGNALS</text><g class="fr-chip"><rect x="804" y="1033" width="63" height="22" rx="11"/><text x="835.5" y="1048" text-anchor="middle">logging</text></g><g class="fr-chip"><rect x="873" y="1033" width="63" height="22" rx="11"/><text x="904.5" y="1048" text-anchor="middle">metrics</text></g><g class="fr-chip"><rect x="804" y="1061" width="90" height="22" rx="11"/><text x="849.0" y="1076" text-anchor="middle">otel export</text></g><g class="fr-chip"><rect x="804" y="1089" width="116" height="22" rx="11"/><text x="862.0" y="1104" text-anchor="middle">langfuse export</text></g><g class="fr-chip"><rect x="804" y="1117" width="76" height="22" rx="11"/><text x="842.0" y="1132" text-anchor="middle">dashboard</text></g></g><g class="fr-node fr-float" style="--i:12"><rect x="30" y="1463" width="300" height="192" rx="14"/><text class="fr-name" x="46" y="1492.0">Vector DB adapters</text><text class="fr-path" x="46" y="1509.0">adapters/vectordb/</text><text class="fr-group" x="46" y="1533.0">CONTRACT</text><g class="fr-chip"><rect x="44" y="1540" width="63" height="22" rx="11"/><text x="75.5" y="1555" text-anchor="middle">base.py</text></g><g class="fr-chip"><rect x="113" y="1540" width="83" height="22" rx="11"/><text x="154.5" y="1555" text-anchor="middle">factory.py</text></g><text class="fr-group" x="46" y="1582.0">BUILT IN</text><g class="fr-chip"><rect x="44" y="1589" width="56" height="22" rx="11"/><text x="72.0" y="1604" text-anchor="middle">qdrant</text></g><g class="fr-chip"><rect x="106" y="1589" width="70" height="22" rx="11"/><text x="141.0" y="1604" text-anchor="middle">pgvector</text></g><g class="fr-chip"><rect x="182" y="1589" width="56" height="22" rx="11"/><text x="210.0" y="1604" text-anchor="middle">chroma</text></g><g class="fr-chip"><rect x="244" y="1589" width="56" height="22" rx="11"/><text x="272.0" y="1604" text-anchor="middle">milvus</text></g><g class="fr-chip"><rect x="44" y="1617" width="70" height="22" rx="11"/><text x="79.0" y="1632" text-anchor="middle">weaviate</text></g></g><g class="fr-node fr-float" style="--i:13"><rect x="350" y="1463" width="300" height="192" rx="14"/><text class="fr-name" x="366" y="1492.0">Embedding adapters</text><text class="fr-path" x="366" y="1509.0">adapters/embeddings/</text><text class="fr-group" x="366" y="1533.0">CONTRACT</text><g class="fr-chip"><rect x="364" y="1540" width="63" height="22" rx="11"/><text x="395.5" y="1555" text-anchor="middle">base.py</text></g><g class="fr-chip"><rect x="433" y="1540" width="83" height="22" rx="11"/><text x="474.5" y="1555" text-anchor="middle">factory.py</text></g><g class="fr-chip"><rect x="522" y="1540" width="83" height="22" rx="11"/><text x="563.5" y="1555" text-anchor="middle">tiering.py</text></g><text class="fr-group" x="366" y="1582.0">BUILT IN</text><g class="fr-chip"><rect x="364" y="1589" width="90" height="22" rx="11"/><text x="409.0" y="1604" text-anchor="middle">huggingface</text></g><g class="fr-chip"><rect x="460" y="1589" width="56" height="22" rx="11"/><text x="488.0" y="1604" text-anchor="middle">openai</text></g><g class="fr-chip"><rect x="522" y="1589" width="56" height="22" rx="11"/><text x="550.0" y="1604" text-anchor="middle">cohere</text></g><g class="fr-chip"><rect x="364" y="1617" width="56" height="22" rx="11"/><text x="392.0" y="1632" text-anchor="middle">ollama</text></g></g><g class="fr-node fr-float" style="--i:14"><rect x="670" y="1463" width="300" height="192" rx="14"/><text class="fr-name" x="686" y="1492.0">LLM adapters</text><text class="fr-path" x="954" y="1491.0" text-anchor="end">adapters/llm/</text><text class="fr-group" x="686" y="1517.0">CONTRACT</text><g class="fr-chip"><rect x="684" y="1524" width="63" height="22" rx="11"/><text x="715.5" y="1539" text-anchor="middle">base.py</text></g><g class="fr-chip"><rect x="753" y="1524" width="83" height="22" rx="11"/><text x="794.5" y="1539" text-anchor="middle">factory.py</text></g><text class="fr-group" x="686" y="1566.0">BUILT IN</text><g class="fr-chip"><rect x="684" y="1573" width="56" height="22" rx="11"/><text x="712.0" y="1588" text-anchor="middle">openai</text></g><g class="fr-chip"><rect x="746" y="1573" width="130" height="22" rx="11"/><text x="811.0" y="1588" text-anchor="middle">openai_compatible</text></g><g class="fr-chip"><rect x="684" y="1601" width="76" height="22" rx="11"/><text x="722.0" y="1616" text-anchor="middle">anthropic</text></g><g class="fr-chip"><rect x="766" y="1601" width="56" height="22" rx="11"/><text x="794.0" y="1616" text-anchor="middle">cohere</text></g><g class="fr-chip"><rect x="828" y="1601" width="56" height="22" rx="11"/><text x="856.0" y="1616" text-anchor="middle">ollama</text></g></g><g class="fr-node fr-node--outside fr-float" style="--i:15"><rect x="30" y="1727" width="300" height="72" rx="14"/><text class="fr-name" x="46" y="1756.0">Vector stores</text><text class="fr-detail" x="46" y="1779.0">Qdrant · pgvector · Chroma · Milvus · Weaviate</text></g><g class="fr-node fr-node--outside fr-float" style="--i:16"><rect x="350" y="1727" width="300" height="72" rx="14"/><text class="fr-name" x="366" y="1756.0">Embedding models</text><text class="fr-detail" x="366" y="1779.0">local sentence-transformers or a hosted API</text></g><g class="fr-node fr-node--outside fr-float" style="--i:17"><rect x="670" y="1727" width="300" height="72" rx="14"/><text class="fr-name" x="686" y="1756.0">LLM providers</text><text class="fr-detail" x="686" y="1779.0">hosted, local, or any OpenAI-compatible URL</text></g><g class="fr-node fr-node--satellite fr-float" style="--i:18"><rect x="30" y="1871" width="220" height="155" rx="14"/><text class="fr-name" x="46" y="1900.0">tests/</text><text class="fr-group" x="46" y="1925.0">TIERS</text><g class="fr-chip"><rect x="44" y="1932" width="43" height="22" rx="11"/><text x="65.5" y="1947" text-anchor="middle">unit</text></g><g class="fr-chip"><rect x="93" y="1932" width="70" height="22" rx="11"/><text x="128.0" y="1947" text-anchor="middle">contract</text></g><g class="fr-chip"><rect x="44" y="1960" width="90" height="22" rx="11"/><text x="89.0" y="1975" text-anchor="middle">integration</text></g><g class="fr-chip"><rect x="140" y="1960" width="43" height="22" rx="11"/><text x="161.5" y="1975" text-anchor="middle">eval</text></g><g class="fr-chip"><rect x="44" y="1988" width="50" height="22" rx="11"/><text x="69.0" y="2003" text-anchor="middle">chaos</text></g><g class="fr-chip"><rect x="100" y="1988" width="43" height="22" rx="11"/><text x="121.5" y="2003" text-anchor="middle">load</text></g><g class="fr-chip"><rect x="149" y="1988" width="43" height="22" rx="11"/><text x="170.5" y="2003" text-anchor="middle">soak</text></g></g><g class="fr-node fr-node--satellite fr-float" style="--i:19"><rect x="270" y="1871" width="220" height="155" rx="14"/><text class="fr-name" x="286" y="1900.0">docs/</text><text class="fr-group" x="286" y="1925.0">KEPT TRUE BY GATES</text><g class="fr-chip"><rect x="284" y="1932" width="76" height="22" rx="11"/><text x="322.0" y="1947" text-anchor="middle">reference</text></g><g class="fr-chip"><rect x="366" y="1932" width="43" height="22" rx="11"/><text x="387.5" y="1947" text-anchor="middle">ADRs</text></g><g class="fr-chip"><rect x="284" y="1960" width="123" height="22" rx="11"/><text x="345.5" y="1975" text-anchor="middle">benchmark ledger</text></g><g class="fr-chip"><rect x="284" y="1988" width="90" height="22" rx="11"/><text x="329.0" y="2003" text-anchor="middle">task ledger</text></g></g><g class="fr-node fr-node--satellite fr-float" style="--i:20"><rect x="510" y="1871" width="220" height="155" rx="14"/><text class="fr-name" x="526" y="1900.0">scripts/</text><text class="fr-group" x="526" y="1925.0">GATES</text><g class="fr-chip"><rect x="524" y="1932" width="76" height="22" rx="11"/><text x="562.0" y="1947" text-anchor="middle">doc truth</text></g><g class="fr-chip"><rect x="606" y="1932" width="56" height="22" rx="11"/><text x="634.0" y="1947" text-anchor="middle">claims</text></g><g class="fr-chip"><rect x="524" y="1960" width="110" height="22" rx="11"/><text x="579.0" y="1975" text-anchor="middle">error taxonomy</text></g><g class="fr-chip"><rect x="524" y="1988" width="103" height="22" rx="11"/><text x="575.5" y="2003" text-anchor="middle">openapi drift</text></g><g class="fr-chip"><rect x="633" y="1988" width="70" height="22" rx="11"/><text x="668.0" y="2003" text-anchor="middle">must-run</text></g></g><g class="fr-node fr-node--satellite fr-float" style="--i:21"><rect x="750" y="1871" width="220" height="155" rx="14"/><text class="fr-name" x="766" y="1900.0">deploy/ · .github/</text><text class="fr-group" x="766" y="1925.0">SHIPPING</text><g class="fr-chip"><rect x="764" y="1932" width="83" height="22" rx="11"/><text x="805.5" y="1947" text-anchor="middle">Dockerfile</text></g><g class="fr-chip"><rect x="764" y="1960" width="123" height="22" rx="11"/><text x="825.5" y="1975" text-anchor="middle">compose profiles</text></g><g class="fr-chip"><rect x="893" y="1960" width="29" height="22" rx="11"/><text x="907.5" y="1975" text-anchor="middle">ci</text></g><g class="fr-chip"><rect x="764" y="1988" width="63" height="22" rx="11"/><text x="795.5" y="2003" text-anchor="middle">nightly</text></g><g class="fr-chip"><rect x="833" y="1988" width="70" height="22" rx="11"/><text x="868.0" y="2003" text-anchor="middle">security</text></g></g><g class="fr-pk fr-pk--in"><rect x="-7" y="-3" width="14" height="6" rx="3"/><animateMotion dur="9.0s" begin="-0.00s" repeatCount="indefinite" path="M148.0 106.0 V884.0 C148.0 924.0 205.0 916.0 205.0 956.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--in"><rect x="-7" y="-3" width="14" height="6" rx="3"/><animateMotion dur="9.0s" begin="-2.25s" repeatCount="indefinite" path="M148.0 106.0 V884.0 C148.0 924.0 205.0 916.0 205.0 956.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--in"><rect x="-7" y="-3" width="14" height="6" rx="3"/><animateMotion dur="9.0s" begin="-4.50s" repeatCount="indefinite" path="M148.0 106.0 V884.0 C148.0 924.0 205.0 916.0 205.0 956.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--in"><rect x="-7" y="-3" width="14" height="6" rx="3"/><animateMotion dur="9.0s" begin="-6.75s" repeatCount="indefinite" path="M148.0 106.0 V884.0 C148.0 924.0 205.0 916.0 205.0 956.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--in"><rect x="-7" y="-3" width="14" height="6" rx="3"/><animateMotion dur="4.2s" begin="-0.00s" repeatCount="indefinite" path="M350 1391.0 C350 1431.0 470 1423.0 470 1463.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--in"><rect x="-7" y="-3" width="14" height="6" rx="3"/><animateMotion dur="4.2s" begin="-2.10s" repeatCount="indefinite" path="M350 1391.0 C350 1431.0 470 1423.0 470 1463.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--q"><circle r="4.2"/><animateMotion dur="7.4s" begin="-0.00s" repeatCount="indefinite" path="M652.0 106.0 V884.0 C652.0 930.0 392.0 894.0 392.0 940.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--q"><circle r="4.2"/><animateMotion dur="8.6s" begin="-0.00s" repeatCount="indefinite" path="M652.0 106.0 V884.0 C652.0 930.0 392.0 894.0 392.0 940.0 V1395.0 C392.0 1435.0 290 1423.0 290 1463.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--q"><circle r="4.2"/><animateMotion dur="8.6s" begin="-4.30s" repeatCount="indefinite" path="M652.0 106.0 V884.0 C652.0 930.0 392.0 894.0 392.0 940.0 V1395.0 C392.0 1435.0 290 1423.0 290 1463.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--q"><circle r="4.2"/><animateMotion dur="8.0s" begin="-0.00s" repeatCount="indefinite" path="M652.0 106.0 V884.0 C652.0 930.0 780.0 894.0 780.0 940.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--q"><circle r="4.2"/><animateMotion dur="8.0s" begin="-4.00s" repeatCount="indefinite" path="M652.0 106.0 V884.0 C652.0 930.0 780.0 894.0 780.0 940.0 V1655.0 V1761.0"/></g><g class="fr-pk fr-pk--n"><circle r="4.2"/><animateMotion dur="5.0s" begin="-0.00s" repeatCount="indefinite" path="M400.0 106.0 V668.0"/></g></svg>
</div>
<div class="fr-legend">
<span><i class="fr-key--surface"></i>Entry surface</span>
<span><i></i>Package inside <code>src/fasterrag/</code></span>
<span><i class="fr-key--rail"></i>Used by every layer</span>
<span><i class="fr-key--outside"></i>External system</span>
<span><i class="fr-key--satellite"></i>Repository, outside the package</span>
</div>
</div>

**How to read it.** The three surfaces at the top are deliberately thin. The REST routers, the CLI
commands and the Python facade each build the *same* services from the *same* validated settings, so
a query behaves identically — same retrieval, same errors, same trace — whichever way it arrived.
Nothing in the engine knows which vector database or model provider is configured, which is what
makes a vendor a line in `config.yaml` and not a code change.

## A document goes in

Eight stages, each owned by one module. The line under each stage is what happens when that stage
goes wrong — the part most pipelines leave out.

<ol class="fr-track fr-track--in fr-track--8"><li style="--i:0"><span class="fr-step">01</span><strong>Accept</strong><code>services/ingestion.py</code><p>A job is recorded in the journal and its id returned at once. An idempotency key replayed returns the same job.</p><p class="fr-guard">A bad source is refused here, before any work.</p></li><li style="--i:1"><span class="fr-step">02</span><strong>Resolve</strong><code>services/sources.py</code><p>Paths, URLs and inline payloads become readable bytes. Paths can be confined to allowed roots; URLs are fetched through an SSRF guard.</p><p class="fr-guard">A source that cannot be read fails the request, not the corpus.</p></li><li style="--i:2"><span class="fr-step">03</span><strong>Parse</strong><code>core/parsing/</code><p>A worker process turns the file into structured blocks: headings, paragraphs, tables, code.</p><p class="fr-guard">A corrupt file costs one document. It goes to the dead-letter queue with a reason code.</p></li><li style="--i:3"><span class="fr-step">04</span><strong>Chunk</strong><code>core/chunking/</code><p>Blocks are packed into chunks inside a token budget. Each chunk gets a deterministic id.</p><p class="fr-guard">Re-ingesting the same content produces the same ids, so a retry overwrites.</p></li><li style="--i:4"><span class="fr-step">05</span><strong>Queue</strong><code>workers/queues.py</code><p>Chunks cross a bounded queue. When embedding is slower than parsing, the parser waits.</p><p class="fr-guard">Memory stays bounded however large the corpus.</p></li><li style="--i:5"><span class="fr-step">06</span><strong>Embed</strong><code>workers/embed_pool.py</code><p>Workers batch chunks and call the embedding adapter behind a circuit breaker.</p><p class="fr-guard">Retryable provider errors are retried with backoff; the rest fail only their own document.</p></li><li style="--i:6"><span class="fr-step">07</span><strong>Index</strong><code>workers/indexer.py</code><p>Dense vectors and BM25 sparse vectors are written together through the vector-database adapter.</p><p class="fr-guard">A refused write is retried, then recorded against the document.</p></li><li style="--i:7"><span class="fr-step">08</span><strong>Commit</strong><code>workers/ledger.py</code><p>When the last chunk of a document is written, the journal records it as indexed and the checkpoint advances.</p><p class="fr-guard">A document with any failed chunk is never recorded as indexed.</p></li></ol>

Two properties fall out of this shape. **A job always ends**: an error settles it in a terminal
status and never leaves it `running`. And **the journal never lies about the
index**: a document's dedup hash and its `indexed` record are written after its chunks, not before.

!!! note "What is not built yet"
    A job interrupted by the process exiting is not resumed automatically (TASK-0436), a document
    cannot yet be deleted or updated in place (TASK-0443), and ingestion runs inside the API
    process — the separate `worker` process does no work yet (TASK-0440).

## A question comes out

Nine stages. Retrieval is hybrid by default — meaning and keywords are searched together — and every
stage that can fail has a defined, visible fallback.

<ol class="fr-track fr-track--q fr-track--9"><li style="--i:0"><span class="fr-step">01</span><strong>Ask</strong><code>api/query.py · cli · facade</code><p>A question arrives with an optional collection, filter and top-k.</p><p class="fr-guard">Validated at the boundary; errors are problem+json with a stable code.</p></li><li style="--i:1"><span class="fr-step">02</span><strong>Cache</strong><code>core/cache/semantic.py</code><p>A paraphrase of an earlier question can be answered from the semantic cache, scoped by collection and filter.</p><p class="fr-guard">A miss falls through to retrieval.</p></li><li style="--i:2"><span class="fr-step">03</span><strong>Embed</strong><code>adapters/embeddings/</code><p>The question is embedded with the configured embedding model.</p><p class="fr-guard">Behind the embeddings circuit breaker.</p></li><li style="--i:3"><span class="fr-step">04</span><strong>Search</strong><code>services/querying.py</code><p>Two searches run at once: dense vectors and BM25. Both receive the same pushed-down filter.</p><p class="fr-guard">If the keyword leg has no terms it is skipped and the dense leg answers.</p></li><li style="--i:4"><span class="fr-step">05</span><strong>Fuse</strong><code>core/retrieval/fusion.py</code><p>Reciprocal Rank Fusion merges the two rankings into one shortlist.</p><p class="fr-guard">Each result keeps the rank and score of both legs.</p></li><li style="--i:5"><span class="fr-step">06</span><strong>Rerank</strong><code>core/rerank/</code><p>A cross-encoder reads the question and each chunk together and reorders the shortlist.</p><p class="fr-guard">If the reranker fails, the fused order is used and the degradation is recorded.</p></li><li style="--i:6"><span class="fr-step">07</span><strong>Assemble</strong><code>core/context.py</code><p>Chunks are packed into the prompt inside a token budget; near-duplicates are dropped.</p><p class="fr-guard">Chunks that do not fit the budget are left out.</p></li><li style="--i:7"><span class="fr-step">08</span><strong>Generate</strong><code>services/generation.py</code><p>The LLM adapter answers, streamed as server-sent events, after the cost budget is charged.</p><p class="fr-guard">Without streaming, a provider failure degrades to an extractive answer from the retrieved chunks; the streamed path does not yet (TASK-0438).</p></li><li style="--i:8"><span class="fr-step">09</span><strong>Cite and trace</strong><code>core/generation.py · services/traces.py</code><p>Citation markers are resolved against the chunks that were supplied, and the whole query is stored as a trace.</p><p class="fr-guard">A stored trace can be replayed later under a different configuration.</p></li></ol>

The same path serves `POST /v1/query`, `fasterrag query` and `FasterRag.query`. `POST /v1/retrieve`
stops after reranking and returns the chunks with the rank and score each stage gave them.

## What holds it together

Six mechanisms do most of the work of making the system dependable. Each is one module you can read.

<div class="fr-mech"><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-w" style="--i:0" x="8" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:1" x="25" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:2" x="42" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:3" x="59" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:4" x="76" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:5" x="93" y="14" width="12" height="16" rx="3"/></svg><strong>The ledger</strong><code>workers/ledger.py</code><p>A document is recorded as indexed only when the vector database holds every chunk of it. The checkpoint is a watermark over settled documents, so a resumed job never skips one still in flight.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-w" style="--i:0" x="8" y="10" width="15" height="12" rx="3"/><rect class="fr-w" style="--i:1" x="29" y="10" width="15" height="12" rx="3"/><rect class="fr-drop" style="--i:2" x="50" y="10" width="15" height="12" rx="3"/><rect class="fr-w" style="--i:3" x="71" y="10" width="15" height="12" rx="3"/><rect class="fr-w" style="--i:4" x="92" y="10" width="15" height="12" rx="3"/></svg><strong>The dead-letter queue</strong><code>services/journal.py</code><p>One bad document costs one document. It is set aside with a reason code and can be retried on its own; the rest of the job carries on.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><circle class="fr-w fr-w--3" style="--i:0" cx="24" cy="22" r="9"/><circle class="fr-w fr-w--3" style="--i:1" cx="60" cy="22" r="9"/><circle class="fr-w fr-w--3" style="--i:2" cx="96" cy="22" r="9"/><path d="M33 22H51M69 22H87" class="fr-line"/></svg><strong>Circuit breakers</strong><code>core/breaker.py</code><p>One breaker per provider, shared across the process. Repeated failures open it, and calls are refused quickly instead of queueing behind a dead dependency.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-w fr-w--3" style="--i:0" x="14" y="8" width="80" height="7" rx="3.5"/><rect class="fr-w fr-w--3" style="--i:1" x="28" y="19" width="66" height="7" rx="3.5"/><rect class="fr-w fr-w--3" style="--i:2" x="42" y="30" width="52" height="7" rx="3.5"/></svg><strong>The degradation ladder</strong><code>services/degradation.py</code><p>When a stage fails the answer steps down, visibly: full, then fused retrieval without reranking, then an extractive answer from the retrieved chunks.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><path class="fr-line" d="M10 12H44Q56 12 60 22M10 32H44Q56 32 60 22M60 22H82"/><rect class="fr-w" style="--i:2" x="84" y="13" width="26" height="18" rx="5"/><circle class="fr-w" style="--i:0" cx="10" cy="12" r="4"/><circle class="fr-w" style="--i:1" cx="10" cy="32" r="4"/></svg><strong>Idempotency keys</strong><code>services/idempotency.py</code><p>A mutating request replayed with the same key returns the first result. A retry after a timeout cannot start the work twice.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-rail" x="8" y="17" width="104" height="10" rx="5"/><rect class="fr-grow" x="8" y="17" width="80" height="10" rx="5"/><path class="fr-line" d="M88 9V35"/></svg><strong>The cost governor</strong><code>services/budget.py</code><p>A token budget per tenant is charged before the provider is called. A request over the budget is refused, not billed.</p></div></div>

## Every package, module by module

Open a package to see each module and what it is for. Each description is the first line of the
module's own docstring, as of the commit this page was generated from.

??? abstract "api/ — entry surfaces"

    The REST surface. Routers hold no business logic: they validate, call a service and render the result or a problem document.

    | Module | What it does |
    | --- | --- |
    | `api/__init__.py` | FastAPI surface: routers, problem responses, and the application factory. |
    | `api/admin.py` | Administrative endpoints: doctor, provisioning, and preflight estimation. |
    | `api/auth.py` | Control-plane authentication, scopes, and per-key rate limiting. |
    | `api/collections.py` | Collection management endpoints. |
    | `api/dependencies.py` | Shared request-scoped construction for the routers. |
    | `api/health.py` | Liveness and readiness endpoints. |
    | `api/idempotency.py` | `Idempotency-Key` handling for every mutating endpoint (TASK-0318). |
    | `api/ingest.py` | Ingestion endpoints. |
    | `api/limits.py` | Request body size limiting (`security.max_request_mb`). |
    | `api/main.py` | Application factory, lifespan, and middleware. |
    | `api/metrics.py` | The metrics scrape endpoint and the RED middleware behind it. |
    | `api/problems.py` | RFC 9457 `application/problem+json` responses. |
    | `api/query.py` | Query endpoints. |
    | `api/schemas.py` | Request and response models for the REST surface. |
    | `api/traces.py` | Trace inspection and replay endpoints (D8). |

??? abstract "cli/ — entry surfaces"

    The `fasterrag` command. Every command turns its arguments into the same validated settings the API uses.

    | Module | What it does |
    | --- | --- |
    | `cli/__init__.py` | The terminal half of the control plane. |
    | `cli/commands/autopilot.py` | The `autopilot run` command (D6). |
    | `cli/commands/backup.py` | The `backup` and `restore` commands. |
    | `cli/commands/benchmark.py` | The `benchmark` command. |
    | `cli/commands/diagnostics.py` | The commands that inspect an installation: `config validate`, `doctor`, `status`. |
    | `cli/commands/infrastructure.py` | The commands that stand things up or price them out: `provision` and `estimate`. |
    | `cli/commands/pipeline.py` | The commands that move data: `ingest`, `query`, and `index`. |
    | `cli/commands/portability.py` | The commands that move a collection between deployments: `export` and `import` (D11). |
    | `cli/commands/processes.py` | The `serve` and `worker` commands. `worker` health-checks and waits; it consumes no work yet (TASK-0440). |
    | `cli/commands/replay.py` | The `replay` command (D8). |
    | `cli/commands/traces.py` | The `traces` command: list and inspect stored query traces (D8). |
    | `cli/main.py` | The `fasterrag` entry point. |
    | `cli/output.py` | CLI output and exit codes. |
    | `cli/parser.py` | The `fasterrag` argument parser. |
    | `cli/settings.py` | The single way a CLI command turns parsed arguments into validated settings. |
    | `cli/sources.py` | Expansion of directory arguments into the files a job actually ingests. |

??? abstract "facade.py · sync.py · public modules — entry surfaces"

    The importable package: the async facade, its blocking twin, and the parts usable on their own.

    | Module | What it does |
    | --- | --- |
    | `facade.py` | The `FasterRag` facade: fasterRag as a library rather than a service. |
    | `sync.py` | The blocking facade: fasterRag from a script, a notebook, or a WSGI app. |
    | `chunking.py` | Public chunking surface: `from fasterrag.chunking import RecursiveChunker`. |
    | `parsing.py` | Public parsing surface: `from fasterrag.parsing import parse_document`. |
    | `retrieval.py` | Public retrieval surface: `from fasterrag.retrieval import rrf_fuse`. |
    | `rerank.py` | Public reranking surface: `from fasterrag.rerank import CrossEncoderReranker`. |
    | `evals.py` | Public evaluation surface: `from fasterrag.evals import evaluate`. |
    | `__init__.py` | fasterRag — a backend-only Retrieval-Augmented Generation framework. |

??? abstract "services/ — orchestration"

    Use cases. A service owns a lifecycle — a job, a query, a replay — and calls workers, core and adapters to carry it out.

    | Module | What it does |
    | --- | --- |
    | `services/__init__.py` | Use-case orchestration. |
    | `services/archive.py` | Portable archive export (D11, `docs/archive-format.md`). |
    | `services/archive_import.py` | Portable archive import (D11, `docs/archive-format.md` §Import semantics). |
    | `services/autopilot.py` | Eval-driven auto-tuning that only ever suggests (D6). |
    | `services/backfill.py` | Payload backfill for the tenant tag, and the audit that must precede it. |
    | `services/backup.py` | Backup and restore of a whole deployment. |
    | `services/benchmark.py` | The benchmark suite and ledger emitter. |
    | `services/budget.py` | The D9 runtime cost governor: token budgets enforced before a provider is called. |
    | `services/ci_status.py` | Reads the latest CI verdict for a branch from the GitHub Actions API. |
    | `services/degradation.py` | The degradation ladder's registry: which rungs exist, and what serves each one. |
    | `services/doctor.py` | Preflight diagnostics (D10). |
    | `services/estimation.py` | Preflight cost estimation (D9). |
    | `services/evaluation.py` | Running the eval harness against a live collection. |
    | `services/generation.py` | Answer generation: retrieve, assemble, generate, cite. |
    | `services/golden_sets.py` | Building a golden set from a corpus on disk (P4). |
    | `services/grafana.py` | Grafana auto-provisioning as code (`observability.grafana: true`). |
    | `services/idempotency.py` | Durable `Idempotency-Key` records for the mutating REST surface (TASK-0318). |
    | `services/ingestion.py` | Ingestion: accept a job, run the pipeline, own its lifecycle. |
    | `services/journal.py` | Checkpointed, idempotent ingestion state (D3). |
    | `services/langfuse.py` | Langfuse v3 auto-provisioning (`observability.langfuse: true`). |
    | `services/lockfile.py` | The index lockfile and drift detection (D1). |
    | `services/provision_registry.py` | The one table of what fasterRag can provision, and the three verbs each tool answers. |
    | `services/provisioning.py` | Config-driven provisioning of system-managed containers. |
    | `services/querying.py` | Retrieval: hybrid dense and keyword legs, fused. |
    | `services/regression.py` | The retrieval regression gate (D7). |
    | `services/reindex.py` | Zero-downtime reindexing (D2). |
    | `services/replay.py` | Time-travel replay (D8). |
    | `services/sources.py` | Resolving the three documented source types into readable bytes. |
    | `services/tenancy.py` | Tenant scoping for collection names. |
    | `services/traces.py` | Local trace persistence (D8). |

??? abstract "workers/ — engine"

    The ingestion pipeline: a process pool that parses and chunks, a bounded queue, embedding workers, the indexer and the ledger.

    | Module | What it does |
    | --- | --- |
    | `workers/__init__.py` | Parallel execution: the CPU pool, the embedding pool, bounded queues, and the indexer. |
    | `workers/cpu_pool.py` | CPU worker pool: load, parse, and chunk. |
    | `workers/embed_pool.py` | Embedding worker pool: stateful, batched, retryable. |
    | `workers/indexer.py` | Indexer: writes embedded chunks to the vector database. |
    | `workers/ledger.py` | Per-document completion tracking between the parse side and the index side. |
    | `workers/queues.py` | Bounded queues, backpressure, and the payloads that cross the pool boundary. |

??? abstract "core/ — engine"

    The vendor-free library: parsing, chunking, sparse encoding and fusion, reranking, context assembly, prompt contracts, caches, evaluation, the breaker and the rate limiter.

    | Module | What it does |
    | --- | --- |
    | `core/__init__.py` | The RAG pipeline: parsing, chunking, retrieval, reranking, assembly, generation. |
    | `core/breaker.py` | Per-provider circuit breakers (`reliability.circuit_breaker`, docs/reliability.md §3). |
    | `core/cache/embedding.py` | Embedding cache. Implemented, and not yet wired into the pipeline (TASK-0442). |
    | `core/cache/redis.py` | Redis-backed cache store: one cache shared by every process that reaches the same server. |
    | `core/cache/semantic.py` | Semantic response cache: answer a paraphrase from the answer already given. |
    | `core/cache/stats.py` | Cache hit/miss accounting. |
    | `core/cache/store.py` | Cache storage backends. |
    | `core/chunking/enrichment.py` | Contextual enrichment (P2, `docs/prompts.md`). |
    | `core/chunking/fixed.py` | Fixed-size chunker. |
    | `core/chunking/late.py` | Late chunker. |
    | `core/chunking/layout.py` | Layout-aware chunker. |
    | `core/chunking/models.py` | Chunk structures, token counting, and the shared assembler that enforces invariants. |
    | `core/chunking/pooling.py` | The pooling half of late chunking (D-late, `chunking.strategy: late`). |
    | `core/chunking/recursive.py` | Recursive chunker — the default strategy. |
    | `core/chunking/semantic.py` | Semantic chunker. Usable from the library; the ingestion pipeline refuses it for now (TASK-0431). |
    | `core/chunking/tokenizer.py` | The embedding model's own tokenizer, used to make `chunking.chunk_size` mean tokens. |
    | `core/context.py` | Context assembly: pack retrieved chunks into a prompt, with citations. |
    | `core/evals/generator.py` | P4 golden-set generation from a real corpus. |
    | `core/evals/golden.py` | Golden sets: the ground truth retrieval quality is measured against. |
    | `core/evals/harness.py` | The retrieval eval harness. |
    | `core/evals/metrics.py` | Retrieval quality metrics. |
    | `core/faithfulness.py` | P3 faithfulness scoring: how much of an answer its context actually supports. |
    | `core/generation.py` | P1 answer generation: prompt building and citation resolution. |
    | `core/identity.py` | Deterministic identifiers and content hashes. |
    | `core/parsing/docx.py` | DOCX parser. |
    | `core/parsing/html.py` | HTML parser. |
    | `core/parsing/markdown.py` | Markdown parser. |
    | `core/parsing/models.py` | Parsed-document structures shared by every parser. |
    | `core/parsing/options.py` | Tunable parser thresholds, and how configuration reaches a parser. |
    | `core/parsing/pdf.py` | PDF parser with table extraction, layout-based heading inference, and an OCR path. |
    | `core/parsing/plaintext.py` | Plain-text parser. |
    | `core/parsing/tabular.py` | CSV and JSON parsers. |
    | `core/rate_limit.py` | Per-key request limiting, with the window kept wherever the deployment chooses. |
    | `core/rerank/cross_encoder.py` | Cross-encoder reranking of the retrieved shortlist. |
    | `core/retrieval/bm25.py` | BM25 sparse encoding for the keyword retrieval leg. |
    | `core/retrieval/fusion.py` | Reciprocal Rank Fusion. |
    | `core/retrieval/models.py` | Retrieval result types. |
    | `core/tracing.py` | Query traces and the four RAG spans (D8). |

??? abstract "adapters/ — vendor boundary"

    One interface and one factory per kind of vendor. Every vendor exception is translated into the project's error taxonomy here.

    | Module | What it does |
    | --- | --- |
    | `adapters/__init__.py` | Vendor isolation boundary. |
    | `adapters/embeddings/base.py` | Vendor-neutral embedding provider contract. |
    | `adapters/embeddings/cohere.py` | Cohere embedding provider. |
    | `adapters/embeddings/factory.py` | Selects the concrete embedding adapter from configuration. |
    | `adapters/embeddings/huggingface.py` | Local sentence-transformers embedding provider — the default. |
    | `adapters/embeddings/ollama.py` | Ollama embedding provider — a local server, so no credential is required. |
    | `adapters/embeddings/openai.py` | OpenAI embedding provider. |
    | `adapters/embeddings/sync.py` | Bridge from an async embedding adapter to the synchronous chunker interface. |
    | `adapters/embeddings/tiering.py` | Tiered embedding: route document classes to different models. |
    | `adapters/llm/anthropic.py` | Anthropic generation. |
    | `adapters/llm/base.py` | Vendor-neutral LLM provider contract. |
    | `adapters/llm/cohere.py` | Cohere generation. |
    | `adapters/llm/factory.py` | Selects the concrete LLM adapter from configuration. |
    | `adapters/llm/ollama.py` | Ollama generation against a local server, so no credential is required. |
    | `adapters/llm/openai.py` | OpenAI and OpenAI-compatible generation. |
    | `adapters/vectordb/base.py` | Vendor-neutral vector database contract. |
    | `adapters/vectordb/chroma.py` | Chroma adapter — the contract proved against an in-process, file-backed store. |
    | `adapters/vectordb/factory.py` | Selects the concrete vector database adapter from configuration. |
    | `adapters/vectordb/milvus.py` | Milvus adapter — the contract proved against a distributed, schema-first store. |
    | `adapters/vectordb/pgvector.py` | pgvector adapter — the vendor-neutral contract proved against a SQL backend. |
    | `adapters/vectordb/qdrant.py` | Qdrant reference adapter. |
    | `adapters/vectordb/weaviate.py` | Weaviate adapter — the contract proved against a store with its own opinion of search. |

??? abstract "config/ · errors.py · observability/ — foundations"

    The validated configuration, the typed error hierarchy, and logs, metrics, traces and the read-only dashboard.

    | Module | What it does |
    | --- | --- |
    | `config/__init__.py` | Configuration schema and loader. |
    | `config/loader.py` | Fail-fast configuration loader. |
    | `config/schema.py` | Pydantic models mirroring `docs/config-reference.md`. |
    | `config/template.py` | Access to the canonical `config.yaml` that ships inside the package. |
    | `errors.py` | Typed error taxonomy shared by the API, the CLI, and the Python package. |
    | `observability/__init__.py` | Observability: correlation ids, structured logging, spans, metrics, and the dashboard. |
    | `observability/dashboard.py` | The read-only observability dashboard (`observability.dashboard: true`). |
    | `observability/langfuse_export.py` | Exporting query traces to a provisioned Langfuse instance. |
    | `observability/logging.py` | Structured JSON logging and request correlation ids. |
    | `observability/metrics.py` | The metrics catalogue of `docs/observability.md` §2. |
    | `observability/otel_export.py` | OTLP export of query traces (`observability.otel`). |
    | `observability/otel_metrics.py` | OTLP export of the metric catalogue (`observability.otel`). |


Around the package: `tests/` (unit, contract, integration, eval, chaos, load and soak tiers),
`docs/` (this site, the decision records, the benchmark ledger), `scripts/` (the gates that fail the
build when a document and the code disagree), and `deploy/` with `.github/` (compose profiles and
the CI, nightly and security workflows). [Repository structure](structure.md) has the full tree.

## One file decides the vendors

```yaml
vector_db:
  provider: qdrant        # or pgvector, chroma, milvus, weaviate
embeddings:
  provider: huggingface   # or openai, cohere, ollama
llm:
  provider: anthropic     # or openai, openai_compatible, cohere, ollama
```

`config.yaml` holds every setting and no secret; credentials live in `.env` and are referenced by
variable name. The file is validated in full at load: an unknown key, a wrong type or a missing
referenced variable stops the process before it serves anything. The
[configuration reference](config-reference.md) lists every key and the [cookbook](cookbook.md) has
worked setups.

## What makes it different

**Everything is pluggable through configuration.** Qdrant is the reference vector database;
pgvector, Chroma, Milvus and Weaviate are registered alongside it, selected by a single
`config.yaml` key. The same is true of embedding and LLM providers. No application code changes
when you switch.

**Hybrid retrieval by default.** Dense vectors and BM25 sparse vectors are searched together and
fused, with optional cross-encoder reranking. Both legs receive the same pushed-down filter, so
they always search the same subset of the corpus.

**Blue/green re-embedding.** Changing an embedding model builds a new collection behind an alias
and flips only when an eval gate passes, so a model change cannot silently degrade retrieval.

**A typed error taxonomy, enforced.** No bare `except`, no silently swallowed exception. API
errors are RFC 9457 `application/problem+json` with a stable machine-readable code. A CI gate
parses the source tree and fails the build on a handler that hides a failure.

**Claims are gated, not asserted.** Every performance statement in this documentation is either
a goal or it links to an entry in the [benchmark ledger](benchmarks.md). A CI gate greps for
superlatives and fails on any that do not.

<figure markdown>
  ![The fasterrag command line, showing every top-level subcommand](assets/help.svg)
  <figcaption>Real output from <code>fasterrag --help</code>, generated from a captured run.</figcaption>
</figure>

## Honest status

This is a **beta in its build phase**, and the documentation is deliberately candid about what
that means:

- The [benchmark ledger](benchmarks.md) holds six entries and **marks all six non-citable**,
  because they were measured on non-isolated hardware. The project targets high ingestion and
  retrieval throughput; it does not yet claim it, and the ledger explains exactly why.
- [Release readiness](release-readiness.md) is re-derived at each release decision and published
  whatever it says.
- [Failure modes](failure-modes.md) is a full FMEA table, and [reliability](reliability.md)
  documents the error taxonomy and its enforcement.
- The [verification procedure](verification.md) is a tiered acceptance process for proving a
  checkout matches these documents.

If you are evaluating this project, those four pages will tell you more than any feature list.

## Where to start

| You want to | Read |
| --- | --- |
| Run a query end to end | [Quickstart](quickstart.md) |
| Understand the moving parts | [Architecture](architecture.md) |
| Configure it for your stack | [Configuration reference](cookbook.md) |
| Call it over HTTP | [REST API reference](api-reference.md) |
| Drive it from a terminal | [CLI reference](cli-reference.md) |
| Import it as a library | [Python package surface](python-api.md) |
| Move from another framework | [Migration guide](migration-guide.md) |
| Deploy and operate it | [Deployment](deployment.md) and [Security](security.md) |

## Operational tooling

The CLI covers the whole lifecycle, not just querying: provisioning, preflight diagnostics, cost
estimation, portable export and import, trace inspection and query replay, backup and restore,
and a continuous-integration status check.

<figure markdown>
  ![Collection management subcommands including tenants and backfill-tenant](assets/index.svg)
  <figcaption>Collection management, including multi-tenant payload tooling.</figcaption>
</figure>

<figure markdown>
  ![The ci-status command, which fails when any workflow's last conclusion on a branch is red](assets/cistatus.svg)
  <figcaption>A workflow that is green on push can be red on a schedule; this reads every one.</figcaption>
</figure>

## Licence and source

Apache-2.0. The source, the issue tracker and the full commit history are at
[github.com/balrajsingh165/fasterrag](https://github.com/balrajsingh165/fasterrag).
Contributions are welcome — [CONTRIBUTING](CONTRIBUTING.md) sets out the rules, which are
stricter than most and exist for reasons the [decision records](adr/ADR-0001-qdrant-as-reference-vector-db.md)
explain.
