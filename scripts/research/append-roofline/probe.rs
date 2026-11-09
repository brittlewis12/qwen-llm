//! FAMILY MODEL CORPUS CONTROLS_METAL. Isolated diagnostic installation only.
use objc2_foundation::NSString;
use objc2_metal::*;
use qwen_llm::{append_research as research, gguf::GgufFile, metal::*, tokenizer::Tokenizer};
use serde_json::{Value,json};
use std::time::Instant;
fn emit(x:Value){println!("{x}");}
fn hash(x:&[f32])->String{assert!(x.iter().all(|v|v.is_finite()));blake3::hash(bytemuck::cast_slice(x)).to_hex().to_string()}
fn size(n:usize)->MTLSize{MTLSize{width:n,height:1,depth:1}}
fn gpu(ctx:&MetalContext,f:impl FnOnce(&KernelEncoder))->f64{
    let c=ctx.queue.commandBuffer().unwrap();let e=KernelEncoder::begin(&c);f(&e);e.end();c.commit();wait_completed(&c).unwrap();
    (c.GPUEndTime()-c.GPUStartTime())*1e3
}
fn controls(ctx:&mut MetalContext,path:&str,calls:usize)->Result<(),Box<dyn std::error::Error>>{
    let original=ctx.library.clone();let opts=MTLCompileOptions::new();opts.setLanguageVersion(MTLLanguageVersion::Version3_1);opts.setFastMathEnabled(true);
    ctx.library=ctx.device.newLibraryWithSource_options_error(&NSString::from_str(&std::fs::read_to_string(path)?),Some(&opts)).map_err(|e|e.to_string())?;
    let read=ctx.pipeline("roof4_read")?;let noop=ctx.pipeline("roof4_noop")?;ctx.library=original;
    let input=MetalTensor::zeros_f32(ctx,vec![268435456])?;let output=MetalTensor::zeros_f32(ctx,vec![1048576])?;
    gpu(ctx,|e|encode_fill_f32(ctx,e,&input,1.).unwrap());
    for rep in 0..9{
        let ms=gpu(ctx,|e|{e.set_pipeline(&read);e.set_tensor(0,&input);e.set_tensor(1,&output);e.dispatch(size(16384),size(256));});
        let sums=unsafe{std::slice::from_raw_parts(output.buffer.contents().as_ptr().cast::<f32>(),131072)};assert!(sums.iter().all(|&v|v==2048.));
        emit(json!({"kind":"bandwidth","rep":rep,"gpu_ms":ms,"gb_s":1073741824f64/(ms*1e6),"checked":true}));
    }
    for rep in 0..6{
        let mut counts=vec![48,192,768,calls];if rep%2!=0{counts.reverse();}
        for n in counts{
            let before=unsafe{*output.buffer.contents().as_ptr().cast::<u32>()};
            let ms=gpu(ctx,|e|{e.set_pipeline(&noop);e.set_tensor(0,&output);e.set_bytes(1,&[1u32,1]);for _ in 0..n{e.dispatch(size(1),size(32));}});
            let after=unsafe{*output.buffer.contents().as_ptr().cast::<u32>()};assert_eq!(after.wrapping_sub(before),n as u32);
            emit(json!({"kind":"launch_floor","rep":rep,"calls":n,"gpu_ms":ms,"checked":true}));
        }
    }
    Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>>{
    let a:Vec<_>=std::env::args().collect();let family=&a[1];let g=GgufFile::open(&a[2])?;let tokenizer=Tokenizer::from_gguf(&g)?;
    let text=std::fs::read_to_string(&a[3])?.replace("<|","‹|").replace("<｜","‹｜");let ids:Vec<u32>=tokenizer.encode(&text,false)?.into_iter().map(|v|u32::try_from(v).unwrap()).collect();
    let depth=65536usize;assert!(ids.len()>depth+2048);let mut ctx=MetalContext::new()?;
    emit(json!({"kind":"identity","family":family,"depth":depth,"token_ids_hash":blake3::hash(bytemuck::cast_slice(&ids)).to_hex().to_string(),"source_commit":"2bad80c4416b22ce4821d99c6239cb05f82f173e"}));
    let mut max_calls=0;
    if family=="fn"{
        use qwen_llm::{qwen4exp::Qwen4ExpConfig,qwen4exp_runtime::{Qwen4ExpLoadedModel,Qwen4ExpSessionCapacity}};
        let config=Qwen4ExpConfig::from_gguf(&g)?;let cap=Qwen4ExpSessionCapacity::for_forward_limit(&config,depth+4096)?;
        let mut m=Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx,&g,cap,depth+4096)?;
        emit(json!({"kind":"inventory","tensors":m.append_research_inventory()}));let mut r=m.create_runner(&ctx)?;
        let start=Instant::now();r.prefill(&ids[..depth])?;emit(json!({"kind":"warm_prefix","wall_ms":start.elapsed().as_secs_f64()*1e3}));let entry=r.capture_snapshot()?;
        for n in [400,512,2000,2048]{
            let input=&ids[depth..depth+n];let mut expected=String::new();
            for rep in 0..4{
                r.restore_snapshot(&entry)?;let start=Instant::now();let v=r.prefill_continuation_with_command_checkpoint(input,||Ok(()))?.to_vec();let wall=start.elapsed().as_secs_f64()*1e3;let h=hash(&v);
                if rep==0{expected=h.clone();}else{assert_eq!(h,expected);}
                let t=r.last_prefill_timing().unwrap();
                emit(json!({"kind":"native","rows":n,"rep":rep,"warm":rep==0,"wall_ms":wall,"gpu_ms":t.gpu_ms,"encode_ms":t.encode_cpu_ms,"commands":t.command_count,"hash":h}));
            }
            if n==400||n==2000{
                r.restore_snapshot(&entry)?;research::routes_begin();let v=r.prefill_continuation_with_command_checkpoint(input,||Ok(()))?.to_vec();assert_eq!(hash(&v),expected);
                emit(json!({"kind":"routes","rows":n,"layers":research::routes_finish(n,10),"bitwise":true}));
                r.restore_snapshot(&entry)?;let p=dispatch_profile_begin(&ctx,12000)?;let start=Instant::now();let v=r.prefill_continuation_with_command_checkpoint(input,||Ok(()))?.to_vec();let wall=start.elapsed().as_secs_f64()*1e3;let t=r.last_prefill_timing().unwrap();let p=dispatch_profile_finish(&ctx,p)?;assert_eq!(hash(&v),expected);assert_eq!(p.overflow,0);
                max_calls=max_calls.max(p.rows.len());emit(json!({"kind":"profile","rows":n,"wall_ms":wall,"gpu_ms":t.gpu_ms,"profile":p,"bitwise":true}));
                r.restore_snapshot(&entry)?;let start=Instant::now();let v=r.prefill_continuation_with_command_checkpoint(input,||Ok(()))?.to_vec();assert_eq!(hash(&v),expected);
                emit(json!({"kind":"after","rows":n,"wall_ms":start.elapsed().as_secs_f64()*1e3,"gpu_ms":r.last_prefill_timing().unwrap().gpu_ms,"bitwise":true}));
            }
        }
    }else if family=="glm"{
        use qwen_llm::{glm5_next::{Glm5NextModel,FfnTensors},glm5_next_metal::{Glm5NextWeights,Glm5NextSession}};
        let w=Glm5NextWeights::load(&ctx,&g)?;let model=Glm5NextModel::from_gguf(&g)?;let mut inventory=Vec::new();
        for (i,(b,d)) in w.blocks.iter().zip(&model.blocks).enumerate(){
            let mut actual=Vec::new();let _:Result<_,()>=b.try_map(&mut |t|{actual.push(json!({"shape":t.shape,"dtype":format!("{:?}",t.dtype),"bytes":t.n_bytes()}));Ok(())});let mut source=Vec::new();let _:Result<_,()>=d.try_map(&mut |t|{source.push(t.name.clone());Ok(())});assert_eq!(source.len(),actual.len());
            for (mut t,name) in actual.into_iter().zip(source){let role=format!("{:?}",model.trunk.iter().find(|x|x.tensor.name==name).unwrap().role);t["layer"]=json!(i);t["name"]=json!(name);t["role"]=json!(role);inventory.push(t);}
            let _=matches!(&b.ffn,FfnTensors::Moe(_));
        }
        for (name,t,role) in [("token_embd.weight",&w.embedding,"Embedding"),("output.weight",&w.output,"Head"),("output_norm.weight",&w.output_norm,"Vector")]{inventory.push(json!({"name":name,"shape":t.shape,"dtype":format!("{:?}",t.dtype),"bytes":t.n_bytes(),"role":role}));}
        emit(json!({"kind":"inventory","tensors":inventory}));let mut r=Glm5NextSession::with_prefill_rows(&ctx,&w,depth+4096,2048)?;
        let start=Instant::now();r.prefill_packed(&ctx,&ids[..depth])?;emit(json!({"kind":"warm_prefix","wall_ms":start.elapsed().as_secs_f64()*1e3}));let entry=r.capture_snapshot()?;
        for n in [400,512,2000,2048]{
            let input=&ids[depth..depth+n];let mut expected=String::new();
            for rep in 0..4{
                r.restore_snapshot(&entry)?;research::gpu_clear();let start=Instant::now();let v=r.prefill_packed(&ctx,input)?;let wall=start.elapsed().as_secs_f64()*1e3;let h=hash(&v);if rep==0{expected=h.clone();}else{assert_eq!(h,expected);}
                emit(json!({"kind":"native","rows":n,"rep":rep,"warm":rep==0,"wall_ms":wall,"gpu_ms":research::gpu_times().iter().sum::<f64>(),"commands":research::gpu_times().len(),"hash":h}));
            }
            if n==400||n==2000{
                emit(json!({"kind":"routes","rows":n,"layers":r.append_research_routes(n)}));r.restore_snapshot(&entry)?;research::gpu_clear();let p=dispatch_profile_begin(&ctx,12000)?;let start=Instant::now();let v=r.prefill_packed(&ctx,input)?;let wall=start.elapsed().as_secs_f64()*1e3;let p=dispatch_profile_finish(&ctx,p)?;assert_eq!(hash(&v),expected);assert_eq!(p.overflow,0);
                max_calls=max_calls.max(p.rows.len());emit(json!({"kind":"profile","rows":n,"wall_ms":wall,"gpu_ms":research::gpu_times().iter().sum::<f64>(),"profile":p,"bitwise":true}));
                r.restore_snapshot(&entry)?;research::gpu_clear();let start=Instant::now();let v=r.prefill_packed(&ctx,input)?;assert_eq!(hash(&v),expected);emit(json!({"kind":"after","rows":n,"wall_ms":start.elapsed().as_secs_f64()*1e3,"gpu_ms":research::gpu_times().iter().sum::<f64>(),"bitwise":true}));
            }
        }
    }else{panic!("unknown family")}
    controls(&mut ctx,&a[4],max_calls)?;Ok(())
}
