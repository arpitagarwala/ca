// Only the two public applicability notices needed by the corpus verifier.
export const NOTICE_URLS = [
 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_May_2026_Examination.pdf',
 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_Final_Course_November_2026_Examination.pdf'
];
export async function fetchOfficialNotice(url,fetcher=fetch){
 if(!NOTICE_URLS.includes(url))return Response.json({error:'Notice URL not allowed'}, {status:400});
 const response=await fetcher(url,{redirect:'manual',signal:AbortSignal.timeout(45000),headers:{'User-Agent':'CA-Bhaiya-Source-Checker/5.0 (+https://ca.arpitagarwala.com)'}});
 if(!response.ok)return Response.json({error:'Official notice unavailable',upstreamStatus:response.status},{status:502});
 if(Number(response.headers.get('Content-Length'))>2000000){await response.body?.cancel();return Response.json({error:'Notice exceeds size limit'},{status:502})}
 const reader=response.body.getReader(),parts=[];let size=0;
 while(true){const{value,done}=await reader.read();if(done)break;size+=value.length;if(size>2000000){await reader.cancel();return Response.json({error:'Notice exceeds size limit'},{status:502})}parts.push(value)}
 const bytes=new Uint8Array(size);let offset=0;for(const p of parts){bytes.set(p,offset);offset+=p.length}
 if(new TextDecoder().decode(bytes.subarray(0,5))!=='%PDF-')return Response.json({error:'Official source did not return a PDF'},{status:502});
 return new Response(bytes,{headers:{'Content-Type':'application/pdf','Cache-Control':'no-store'}});
}
