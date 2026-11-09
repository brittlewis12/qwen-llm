#!/usr/bin/env python3
"""Apply explicit diagnostic seams to an isolated source checkout, once."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def replace(path, old, new):
    p=ROOT/path;s=p.read_text()
    assert s.count(old)==1,(path,s.count(old),old[:80])
    p.write_text(s.replace(old,new))

def main():
    shutil.copyfile(HERE/'instrument.rs',ROOT/'crates/qwen-llm/src/append_research.rs')
    shutil.copyfile(HERE/'probe.rs',ROOT/'crates/qwen-llm/examples/append_roofline.rs')
    p=ROOT/'crates/qwen-llm/src/lib.rs';p.write_text(p.read_text()+'\n// Scratch research seam, not a runtime policy.\npub mod append_research;\n')
    replace(Path('crates/qwen-llm/src/qwen4exp_moe.rs'),
            '        end_optional(&mut profile, enc, marker)?;\n        Ok(())\n    }\n\n    fn encode_route(',
            '''        crate::append_research::routes_copy(ctx,enc,layer,&self.views.route_counts,
            [self.weights.routed_gate,self.weights.routed_up,self.weights.routed_down])?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_route(''')
    replace(Path('crates/qwen-llm/src/glm5_next_metal/packed.rs'),
            '        if let Some(recorder) = self.stage_recorder.borrow_mut().as_mut() {\n            recorder.command_gpu_ms +=',
            '''        crate::append_research::gpu_record((command.GPUEndTime()-command.GPUStartTime())*1e3);
        if let Some(recorder) = self.stage_recorder.borrow_mut().as_mut() {
            recorder.command_gpu_ms +=''')
    p=ROOT/'crates/qwen-llm/src/qwen4exp_runtime.rs'
    p.write_text(p.read_text()+'''
impl Qwen4ExpLoadedModel<'_> {
    pub fn append_research_inventory(&self) -> serde_json::Value {
        serde_json::Value::Array(self.weights.tensors().map(|(name,t)|
            serde_json::json!({"name":name,"shape":t.shape,"dtype":format!("{:?}",t.dtype),"bytes":t.n_bytes()})).collect())
    }
}
''')
    p=ROOT/'crates/qwen-llm/src/glm5_next_metal/packed.rs'
    p.write_text(p.read_text()+'''
impl Glm5NextSession<'_> {
    pub fn append_research_routes(&self,rows:usize) -> serde_json::Value {
        use crate::glm5_next::FfnTensors;
        let p=self.packed.as_ref().unwrap();
        serde_json::Value::Array(p.routes.iter().enumerate().filter_map(|(layer,route)| {
            let route=route.as_ref()?;
            let ids=unsafe {std::slice::from_raw_parts(route.ids.buffer.contents().as_ptr().cast::<i32>().add(route.ids.offset as usize/4),rows*8)};
            let mut counts=vec![0usize;288];
            for &id in ids {assert!((0..288).contains(&id));counts[id as usize]+=1;}
            let FfnTensors::Moe(m)=&self.weights.blocks[layer].ffn else {panic!("MoE expected")};
            Some(serde_json::json!({"layer":layer,"counts":counts,"bank_bytes":[m.gate_experts.n_bytes(),m.up_experts.n_bytes(),m.down_experts.n_bytes()]}))
        }).collect())
    }
}
''')
    print('Installed scratch-only route copies and ordinary GLM command timer')

if __name__=='__main__':main()
