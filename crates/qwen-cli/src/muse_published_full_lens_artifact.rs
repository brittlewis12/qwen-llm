use super::muse_full_lens_artifact::MatrixDescriptor;
use super::muse_lens_artifact;
use super::published_pt::{ArchiveLayout, ArchiveSpec, ExtractedPayload};
use anyhow::{Context, Result, ensure};
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerConfig};
use serde::{Deserialize, Serialize};

pub(crate) const SCHEMA: &str = "muse_glimmer.published_full_transport";
pub(crate) const SCHEMA_VERSION: u32 = 1;
pub(crate) const MANIFEST_NAME: &str = "lens.json";
pub(crate) const PAYLOAD_NAME: &str = "transport.f16le";
pub(crate) const PROFILE_NAME: &str = "eyes_ml_muse_glimmer_30b_j_v1";
const MATCHED_J_PROFILE_NAME: &str = "brittlewis12_muse_glimmer_30b_j_pile10k25_v1";
const R_PROFILE_NAME: &str = "brittlewis12_muse_glimmer_30b_r_pile10k25_v1";

const SOURCE_REPOSITORY: &str = "eyes-ml/Muse-Glimmer-30B_jacobian-lens";
const SOURCE_REVISION: &str = "71d8434fbd38c8b5d70e1ff1ff2095d5da926c34";
const SOURCE_FILENAME: &str = "Muse-Glimmer-30B_jacobian_lens.pt";
const SOURCE_BYTES: u64 = 4_518_854_369;
const SOURCE_SHA256: &str = "397dc00807a8f72d9d21feaeedeba7fb9e7c50e5c80efc317f37e8cd9c833300";
const DATA_PICKLE_SHA256: &str = "f187e6221a2c5540769e3af89d4e0ea81159a1c98a7a1d405c22384f7bd605a7";
const SERIALIZATION_ID: &str = "1371680345541892666311410627619629120835";
const ARCHIVE_ROOT: &str = "Muse-Glimmer-30B_jacobian_lens";
const PAYLOAD_BLAKE3: &str = "64f50f387a56a4533631e62789a896e759f0ebbdb3a9fdbae45898a5dc8a2794";
const MATCHED_J_SOURCE_REPOSITORY: &str = "brittlewis12/muse-glimmer-30b-r-lens-checkpoints";
const MATCHED_J_SOURCE_REVISION: &str = "4f73cadc74ab26263f3860888a2c274cce452951";
const MATCHED_J_SOURCE_FILENAME: &str = "muse-glimmer-30b-j-lens.pt";
const MATCHED_J_SOURCE_BYTES: u64 = 4_518_856_730;
const MATCHED_J_SOURCE_SHA256: &str =
    "4d8f0cad7623216df5a4d0632cd07a184fe2fb3c9902811c81c8c122086d7d69";
const MATCHED_J_DATA_PICKLE_SHA256: &str =
    "5d5e86d1628d7cba8e645b415cb05271a02f95fc998a0dd19040678a4fa57972";
const MATCHED_J_SERIALIZATION_ID: &str = "1371680345541892666310359746187447669417";
const MATCHED_J_ARCHIVE_ROOT: &str = ".muse-glimmer-30b-j-lens.pt.tmp";
const MATCHED_J_PAYLOAD_BLAKE3: &str =
    "edd5b07de8a9dcc8897904bebd764f73c03790845062588a4f9dabb481ec19e2";
const R_SOURCE_REPOSITORY: &str = "brittlewis12/muse-glimmer-30b-r-lens-checkpoints";
const R_SOURCE_REVISION: &str = "b406c8465c9a49657e30af07753cd08ae7f96f56";
const R_SOURCE_FILENAME: &str = "muse-glimmer-30b-r-lens.pt";
const R_SOURCE_BYTES: u64 = 4_518_856_794;
const R_SOURCE_SHA256: &str = "3675c1371e0d0e97153364b15b38ab169fedc2a392e80c3e4e2e0586d62b068c";
const R_DATA_PICKLE_SHA256: &str =
    "24879bfdc435edefb27dc989245dcd07ec237a5d5e23448c61f06227a4298a7f";
const R_SERIALIZATION_ID: &str = "1371680345541892666304555400842477132771";
const R_ARCHIVE_ROOT: &str = ".muse-glimmer-30b-r-lens.pt.tmp";
const R_PAYLOAD_BLAKE3: &str = "31f9cbadfb0ab969a18cf8dcf3a48be4070645240189e1135e87d513a0b1acf8";
const IDENTITY_MATRIX_BLAKE3: &str =
    "e29104d17d84e4be7d4cac32ccc8470eb46475733e387c6ad12d6a49feaf9574";
const HIDDEN_SIZE: usize = 6_656;
const SOURCE_LAYER_COUNT: usize = 51;
const TARGET_LAYER: u32 = 51;
const MATRIX_BYTES: u64 = HIDDEN_SIZE as u64 * HIDDEN_SIZE as u64 * 2;
const PAYLOAD_BYTES: u64 = MATRIX_BYTES * SOURCE_LAYER_COUNT as u64;

const MATRIX_BLAKE3: [&str; SOURCE_LAYER_COUNT] = [
    "fa2be6849e83601a404dd6f059e55c2fc716ad681537d3ed7c586fb2fbc4b89c",
    "e62902ac559c99c97a5e6f9f4d7737eb2e44ac282b8ae7e78c7bdd097e1bbe22",
    "b2909e16eb89c3a7953942f9194ef0bb1c7b42b6fbbce2fb7f772a95d0917aea",
    "7da82020557cd9b050c6777ec5f79324423e5d63e8525385c3378d5447d89ea2",
    "d8f85c10d9988f43023b8efe81242b429e9f4c4fda79eb21cadb49dd9c070d0c",
    "f089f65149aff08b6c26e7d7a8b617baebfe0f9212b84f14ac3a4d5ddeabb865",
    "98bdd514b651e7c6f2c2acb02d55e48359d792cc4aa4efc4036b9edbc17d9531",
    "1bbfe43e6a2da8527710b9b850cdae24ca1fb18f6d129f6be4799bd766d3af39",
    "8400445201c043bdb08a95bba3f0254da7155c3b9fa57c43d4a22cf216130cb6",
    "8d7487f34fe099b0c8c1e5877aeb3a73661648e4e2d3f8aab57d074c851190ce",
    "82367272d1b5706b0d7f2251d061a04f35dc303b0ff0494a71118fa76f86600c",
    "45096e86ce9edaff80f68e55f7419e808f576a9631767ae1e14191361641860d",
    "68fed655ec9616adbf171bcbe251661b05a1a1d878fef8390c3387199fe39906",
    "a95ed92955e2490e4660653bde31a19ccc7afe174e5b43cb4fab2ff14fc98b9f",
    "dc43f3fa355dbf65543e3a04e218a7effd05713fd5107dd0b53fd11f41ab5149",
    "171cb9b18b2d5e969e85eaa28970f8f08a7945987c2b5bc699dc726b07f372e3",
    "64ef67d57084615ff9a18179869166a97ff34865efd2b3f0d8212a5c847cd01f",
    "e2ada606a901bc3aa42c26aa74a3fe424663f3574512a2ea2fd57dac19ca35d4",
    "6f06e143023feaafcf6fca7ba11918d8efcb65fb202b9ea4c56ca018f8a4967d",
    "7ae96a62ea8c3e3bed4abfa47ba3f93364ed6fe88728b2dfe02372e2298d7856",
    "1ae281bc8146eff33c5849dd01ab96cf18a13b9ba4f24dfba6e142541a321f17",
    "65044e6625383f23514d7bd188f69a0ff52db2f7dca2a0ddece96382f00af49b",
    "fd864a577466ac8ed8a975e6a52f4286ed675910ab1c72006b80388bbcabb76a",
    "d83a36e7f5e3d5bd3e050c1686a3ac4058041756d708e373fccd6121e97f3811",
    "6709d82cbf805557613f603bccc84af2a0bcedce85d83223136debfc3bc0ff96",
    "7701f2097dbefebcb3758007f907fad067e735d12a9841e704a8f47ea8b5d687",
    "1a8673b3b7214cbab246bfccb27a9611077f6d165926c79ef82a245732ec91ec",
    "99a144a3d2c20d7312016daf2f897abcd4a634d97c8dadb879fff2da29cb78c2",
    "014a14bb0ea0dc7903f5869b7cbb6b053c3b33368ed525d1deebec36a2177790",
    "9b37e96c9ba12a3ede4ef0df3b5f5d95c2e1353614772aa60641439d259c481d",
    "1ee9f7e678f352c8f138acf94a5cd8377faa3370cb328f42b070d6d32b97ca56",
    "ea372ede98df1899c45097460dc1fc32d3a76a5ce077318a6bca568962e7068b",
    "4dbd6cf5afe4229ac41ba88423b2ecf040a737cff015b90b9af89f4f7aa99966",
    "d260ea7f81e987d3a0631784351a972c138694accac1148bcb8f1adf52c1acf8",
    "a618dbbcc58066c0c4ac6be4adf191a1ea928330bdc9d9de42895d9a4634d52d",
    "853946be4070a949e3ddd64467a0f656c33ae49c5c117f0eb714cf9f7176e613",
    "f10c4fbfc608dd3563d9bdc94c28eb6547ed95c3f2c1e97d6d178cb7a6921907",
    "2a9afcaf27bcfc4389f46e36572f0147c0686081abd9e244f785073dcd38078a",
    "0443e63b4750a00715c815bce41a24c0a25c6e53d7d2e480e0ec26946fe37d80",
    "b8fd28883f1ae6d031878574a322e834369e6af1836c8e63f5f0ae76ff5c55ae",
    "aee6279d6d50a460179ab0ec8c968c0c4f07a7c0f67e4f5a4973b258b4a88eac",
    "afd8141e96c77f50d4ae348cd31f888e6cd924d987c106c529d763dc2ab17862",
    "4adef04732bf1273151bce460f055d6010eeef9656a13e005c944ba483e84abc",
    "03d4dab334f3b2f65db0dc84090b21bce113d4a680afcb8801585c708193d05f",
    "ce51bed2f5bdbd39130280c4f894421e5438ac74d88ae0dddb894c40f3983405",
    "5b1e3c045861b6e48b5c2bd5362c264185df0e584fcf225cac5e464e36fed2f2",
    "789a7ecf6bc1edd5305a889f9d53d13b6abda4fd4e12fcb3e5f464aaa676b53d",
    "466750a526bf395f5bd3904a21509b54297aee5da713390be057fd6cb5e4d521",
    "2942d708884d6ccf5f1d38744a1ff988cbcbfe23b286379d0cc6a067f2115157",
    "2cc225637559772b512f9d09fc227049d8e6d376e15c0991bbed47ccb3d23fe7",
    "ca84caa1dbcf0af8bdf9242b56ebb2eba548201cf60b8fad460c5d88b9e36027",
];

const MATCHED_J_MATRIX_BLAKE3: [&str; SOURCE_LAYER_COUNT] = [
    "d4d6f1634c3d7a54e1e4fcea4b304b1a8df894bfef538a1cbdfd78875fd8ee46",
    "00e028eb0f140f437fc5d74f48264d311326402242fb4e8b4062985745cc236d",
    "de373cd04cdcb02baa5e22460678e4123bca9a7dccfe83c3f5e7453f8c15ebee",
    "af0b622a746a0c0a1fb668ed16b7f3c6b894c047d99774b31e23a8ca9331ad52",
    "d12411463dd35795d08cf78303f243663766e276916eeacf6baf13573e54d9dd",
    "c6cf4a7004a4b70975481bca452e169b2073f5d07411fd1940f8b4bdf3245a21",
    "484e7c1457c01d2610f066636b0cfb005433398d5a7a7e93f4d865ab4ecedbe0",
    "4814b199110f74ce181030b848403ec9c1720cad14b7710f55b62ca9d264e97c",
    "82b0f145a10fcfae2693b2ce7f05ac043cefe2426e5274b5ae01639985b5bd2e",
    "10ad6d77ffb527de3b467d03db6a19f86d72c63f0f0207b3c3cf829120324bd2",
    "4c6b9996e22fb940823289a8d71592f7fc050a0e9f4e6411c7f04c2426601e08",
    "5dfd6c6ccae8bbdbb0766f775dcf9834cadc591c29d9d32791108067466874a8",
    "aa256f7c2978ec674f45f43ac6827f4f1c7d38ec8f058cd24c797a8cf3288146",
    "7244a3c4cbd0e89d706ea9bae4b279f1f2b904490d166ecc91e1042eeaa2cab5",
    "bdebfb0f62a46e478bc7b465af0dedc8a0c09b23b7ff52e06979e2da643a09cf",
    "0de6eb204a3f564855c21a7273b7708d7570d9e9f366a8d1e6cfcdc2eb8cdab2",
    "faac874df243cd6288f7f37a24b8be60bf6302ea50a7fd8f6e12b7d893692585",
    "f6e017ea9c3304c96f7333a342a2209f139a1222672f93f9057d40f54c7419a4",
    "d5d572a550790e7cf2c1a379931972ef58e13c78864825f737a07aae96e5c181",
    "003cee8a866d6ccb68e49d8acc355a5e7bde55b18c5cff10d4579f2db43f8069",
    "9264d98d2af3bd931e5fe79889094fba23db75c85ce1f93180cdd4d781ed2a96",
    "6f13c9ab065be8a96d335ad135a3b28eb7cd3825da94b2bdd0a9967b3593edf9",
    "8c7f858ef77f3fb84458eda1e3e9c27d95d4bdfe1fc756c814a9671684c2ba78",
    "627ce3beeefb60690b847052ce2e72edfaf5bd7905c1d38abb3ced24b1fd5b93",
    "ac051a6bcf976b7dc33ceff24443757890396a3f898ad930ce487700841b2ffd",
    "4e871d13861155169598d17c25dd0743ce42e43a6d07c9630af7f827ebd0e023",
    "25282b838738615839d3983060ebd616728bc00ef7bdfc6a175c8c3b36f2f4c7",
    "505e4639d668f8e7d5a37efe51571ca1339feeb077c917cc6ebb34894dc9c329",
    "ae6345003a28d327c670f58523376b9798c6cb66605035dcb0272c7911396d20",
    "a0a8a1adda19a7b7d6c0297804fe58d2c7128a315054025f8112cc988c73588c",
    "93a652feee21021b39fcb486fa96c10fb2b8db17167f8a2bb9a109a7e96ab226",
    "a194396fecce52f74d375b406210dc2f4f9264c2aed9f9b55194fc0c6f0305ba",
    "d512b36d452eef14ab291093396da7ea34bf84703a847edfcf77b79d84861244",
    "8ac62b517df6738fbb441b4f181a4974dc91bcd29ab59190593f1a2ea9e2dca1",
    "dc7e9fcb0769aca82ad51b4a2fcacdb355233160f76ca90d4f87bb31c013243e",
    "66ce68a1bbb75face6ffe72d6e98cb8f935a1479fad69f92c0f64d0fd80f4654",
    "7d48ed5a71f3aaa29af8671d6e9169cf2875ce47ac94ff1dcb1452de54bafe31",
    "636618f9bb1629627e10dbe749e9ebc85af27bdcd0dcfb4afaed2746f7ea757d",
    "d90e53d1505375cf1b2e93c11ffe3b276355c989805db550d7ba95081dbcd2cd",
    "54d7c1bb7cc64f464e969a4976a86abe7db479181c2f3f3577ac17dbd12bc1e8",
    "7245974b440c83d40bcdcf576c56b43eccd5d2138c8f2d44708f9bbc9be76209",
    "769771a2e722ec5042ccf613d98ed0752773f5fbc37e520d38b091e13a3d7fe0",
    "a39d86a30517ccbb965202114b818c6a625354e4c510482c33789f3e58e147cf",
    "c779a3c84203d2c21122389efc5325a89bc2ce748c568e17613fa8249801b3c3",
    "af20efa29e1edc8bb09a6a0a264f5b7d364a75b6beac2fb35aa0774963ab7184",
    "4ef7aa217650b19b78787fdfa00e725bc7fe8eb615e9b5bbfa89f3becd7f6ec8",
    "c0b5871ced9791fddff9b8ae39f76980ef94d4b8329c85e210491dfeb4e36353",
    "ef4a5d267781d26f5c8e9daa4c0ea0c08eed2c01290bbb423b7897ac5da9734b",
    "5ddf3c1d62774c6e221bcc8458abff18581d32be9cdf8b48226c6f41af0ee20f",
    "041309aa9290dfe96af8e49beb1c130f9bace58f1d27f085e7a6007639735678",
    IDENTITY_MATRIX_BLAKE3,
];

const R_MATRIX_BLAKE3: [&str; SOURCE_LAYER_COUNT] = [
    "ebb330b82ddd4c7159d67c6134465463ddbfee662cbce2b6d55e62544a649a32",
    "f342b8fa4bb8ee583ad63da237d1a8fbf0ce7c6678b55a4f5531e3419389ff57",
    "5b9c25e3ec77b5899a5a9ff7f6c4c6fe54a268c9f96189e10e35a44528d398b1",
    "993b99fa7e40e0ecc18e5a58a8b5ffff59e55d49de7b8c466efcb9ea1ff436be",
    "0461b68acb47b5be0007560d74d47bec828f45f13387e09ad1d4f315ee0b301c",
    "2da050cc3067c4686b88a69402f8b21f76e56d48724f35af0707d45ed44c61a5",
    "46be89c23b34b15b9068d42a490961ce5f5107354f5b78f9b0c9bd3259ff46f5",
    "06d3d245d06e8e4eab5eb26a6025d8afcfcdd4af1b87aac93189bb160667619f",
    "e4a1bda242de9e8d3b88194a446dd68dea1f1332f0997e1b335dc721ac9c6bcf",
    "988ccfe7c29504114d249fa4578ec9b6596a8e610065c9952fa969f27c543b14",
    "602c8804fafee29d9dc079de6a118b2d41fffa098a338795d526dcb7ec2c2b67",
    "22f2ceecc7326d4e2598241a8b4d79ee6065db6944d6efc4ef02ab177279799b",
    "435ea91fbfae99dd8b8936402d6560e9e2e74f317ef97dfee7fc638bfe8c3e8d",
    "6ea9a3e7c00cad988acecb51179f69d189476cae4d0497082fc18b72d56edfe1",
    "265e31639439f9d630dbdf2894738d9611a8591ccb090745ebe0de12e9e4f757",
    "9473a24b2723c800017047b2f483d242d347a1102b1b27dc1debf8507f137a0f",
    "ee08300e26ee1afb62d42942b5954b26ce040fbb6b09226fefbf49cc94fc1a99",
    "7672a321c149c8657b849b6cd578f785bd9ed9a1596b60a3ccd8fc22c29d8eb5",
    "1fd71757b093b9e30ccf87d2b54e5d37a0772a98c0e579ce6eb66da1e61c8a4d",
    "a9fa49abd019cd4cad11e1230cf7b40d17adc8e942f68453274194d84ce44fb2",
    "572f110f472535f5f0351b651f92e214ae09d6a751818fed101ec7b49aa05a82",
    "1fcef10654220a301b1da64c150021120f9182bffe09b64eb2c716fcc0f4df65",
    "769c06619fedf1d80b9d1eefc6d7833e5e610a59ea233c5f74ce3ec7498c0999",
    "09504fb56ef3503325f4b9324ee5c5e84b11019de50ea7746d7d141a95f7922e",
    "4871e75f38c02f13d0a86f1288776fff34ef45906b9e95257e6d247274477e05",
    "8793103818f3304dcf5d8dd46044f1fb3a605e21903341377ccd26cb645127a3",
    "e48b68f87d9cfa146ec3a9adb442eb5d8d396a8600c7b6b5e2ef3b9ec7957c5c",
    "abaa062ee12e4828830f3cf8df7aff67058970fa10976d709bd433c399e09980",
    "f2c33820902b5052db96b0399e8ef4b8b27f89f66145c9cc991e7af364097eaa",
    "738ab01008d9371f1af86f243ff46005989ce48babdfca6e2bcd03c1cbb31019",
    "7e192ee002924b5a2ab88b297635d81a1fbf43b5db6f0e5533e67dd240fd3cde",
    "cf8234536ae9d12f4f20db948def6edb7584a24d3c8fd95d07e99a528bf112a7",
    "403b5365836740eb3fd45206f21733177e949835eb2265aca7998244b45c654e",
    "76ca08dfdaa80f4186dae657745aabf9ddd42dd1a593dcf1bdab376dfa4956a8",
    "f37e5d4998ae5b5f1e1dbb2c36e4f48cb415937c85eb5a9664b8cac3f0a89220",
    "a7d18e8faab6a9d4361cb5255d91fc1288981dda24c89ddabe891ab835fbdabe",
    "660c3f4490f172e304509a9f8333a62b9f67de1fba7f057066b35e5b23ede25c",
    "9c981817ee3079e8cef7a8c1813a42f76333f5136804dd5df708aa7a26d0b0ff",
    "6a2f77bb21815693725cc07fe24d758a1a6aee8b9633db3ad1d8ff8c18cb8553",
    "d69dca04573853a60a9e7b51838fe2c69d0e1e455634f10a32a31cb0b3e51b06",
    "21ee7298c64e2fa47cc4384d65f3550d947bd8739be303e67fc48433ef68ce65",
    "b7fabaccb6f68841cfed1331ed5c3c2a08eabb11b749c9c5750d6d46acdb3c12",
    "07000c3e6398d390eb508ab58fa8cb0d9ec42b6236ec171c85f394786dfd184b",
    "e65ad0d8477a257eb58f50d1b8324d560865041dae761354a4e0d8b5ea5e620c",
    "d6c206036ce1826f81c14ba93f74c6def3272a88b42028d8b0edfdd9d26fab29",
    "0ca982ec51ac3c39adc99778ac5e127152efdcdb08f4f0f98f70aed518595d66",
    "adae2ff6c0dc48030334f209588ed15a11aa498721a3c01ceab23425e1401bd6",
    "c975e476f1342326bd6c11e71adf1c6b4aaab4857af239c8dfda735a2f3dabba",
    "888a50e0be1f40417026964aba885412172b9dc73f57094e3800eb7eec961a9b",
    "5fa0c5df711c937bdb8d67effe12a64864cf03acdce89fe229170558403b594f",
    IDENTITY_MATRIX_BLAKE3,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProfileId {
    EyesMlMuseGlimmer30bJ,
    BrittLewisMuseGlimmer30bJ,
    BrittLewisMuseGlimmer30bR,
}

#[derive(Clone, Copy, Debug)]
struct ArchiveClaims {
    root: &'static str,
    layout: ArchiveLayout,
    data_pickle_sha256: &'static str,
    serialization_id: Option<&'static str>,
    identity_layer_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct ModelClaims {
    base_model: &'static str,
    fitted_checkpoint: &'static str,
    fitted_checkpoint_revision: &'static str,
    tokenizer_checkpoint: Option<&'static str>,
    tokenizer_revision: Option<&'static str>,
    output_rmsnorm_epsilon: f64,
    output_multiplier: f64,
    final_logit_softcap: f64,
}

#[derive(Clone, Copy, Debug)]
struct FitClaims {
    claims_basis: &'static str,
    embedded_provenance: bool,
    fitter: &'static str,
    fitter_revision: &'static str,
    transformers_revision: &'static str,
    recipe_id: Option<&'static str>,
    dataset: &'static str,
    dataset_revision: Option<&'static str>,
    split: &'static str,
    corpus_preparation: &'static str,
    corpus_selection: Option<&'static str>,
    corpus_text_sha256: Option<&'static str>,
    corpus_token_ids_sha256: Option<&'static str>,
    estimator_contract: Option<&'static str>,
    arithmetic_contract: Option<&'static str>,
    n_prompts: u64,
    max_sequence_length: u32,
    skip_first: u32,
    valid_positions_per_prompt: Option<u32>,
    dim_batch: u32,
    model_execution_dtype: &'static str,
    serialized_dtype: &'static str,
    stop_rule: &'static str,
    convergence_status: &'static str,
    modality: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct TransferClaims {
    binding: &'static str,
    deployed_checkpoint_policy: &'static str,
    validation_status: &'static str,
    image_token_status: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Profile {
    pub(crate) id: ProfileId,
    name: &'static str,
    method: &'static str,
    rule_contract: &'static str,
    target_layer: u32,
    source_repository: &'static str,
    source_revision: &'static str,
    source_filename: &'static str,
    source_bytes: u64,
    source_sha256: &'static str,
    source_license: &'static str,
    archive: ArchiveClaims,
    expected_payload_blake3: &'static str,
    matrix_blake3: &'static [&'static str],
    model: ModelClaims,
    fit: FitClaims,
    transfer: TransferClaims,
}

const PROFILES: [Profile; 3] = [
    Profile {
        id: ProfileId::EyesMlMuseGlimmer30bJ,
        name: PROFILE_NAME,
        method: "J",
        rule_contract: "published_standard_jacobian_lens_v1",
        target_layer: TARGET_LAYER,
        source_repository: SOURCE_REPOSITORY,
        source_revision: SOURCE_REVISION,
        source_filename: SOURCE_FILENAME,
        source_bytes: SOURCE_BYTES,
        source_sha256: SOURCE_SHA256,
        source_license: "Apache-2.0",
        archive: ArchiveClaims {
            root: ARCHIVE_ROOT,
            layout: ArchiveLayout::LayerStorages,
            data_pickle_sha256: DATA_PICKLE_SHA256,
            serialization_id: Some(SERIALIZATION_ID),
            identity_layer_index: None,
        },
        expected_payload_blake3: PAYLOAD_BLAKE3,
        matrix_blake3: &MATRIX_BLAKE3,
        model: ModelClaims {
            base_model: "meta-models/Muse-Glimmer-30B",
            fitted_checkpoint: "eyes-ml/Muse-Glimmer-30B",
            fitted_checkpoint_revision: "97e6fe0a8d8d221b100cd67f53fccf0744950abf",
            tokenizer_checkpoint: None,
            tokenizer_revision: None,
            output_rmsnorm_epsilon: 1e-5,
            output_multiplier: 0.19611613513818404,
            final_logit_softcap: 20.0,
        },
        fit: FitClaims {
            claims_basis: "pinned_repository_model_card_not_embedded_in_pt",
            embedded_provenance: false,
            fitter: "neuronpedia_utils/jlens/fit_lens.py",
            fitter_revision: "7724688596eb734a0662f911bf183151a5c66b2f",
            transformers_revision: "a61d9a57c1ca1018fd84acabbf2104fdf468e143",
            recipe_id: None,
            dataset: "Salesforce/wikitext:wikitext-103-raw-v1",
            dataset_revision: None,
            split: "train",
            corpus_preparation: "streamed_rechunked_approximately_2000_char_prompts",
            corpus_selection: None,
            corpus_text_sha256: None,
            corpus_token_ids_sha256: None,
            estimator_contract: None,
            arithmetic_contract: None,
            n_prompts: 900,
            max_sequence_length: 128,
            skip_first: 16,
            valid_positions_per_prompt: Some(111),
            dim_batch: 8,
            model_execution_dtype: "bfloat16",
            serialized_dtype: "float16",
            stop_rule: "smoothed_delta_mean_below_1e-3_after_at_least_100_prompts",
            convergence_status: "not_reached_at_900_prompts_final_smoothed_delta_approximately_1.5e-3",
            modality: "text_only",
        },
        transfer: TransferClaims {
            binding: "published_checkpoint_geometry_transfer",
            deployed_checkpoint_policy: "supported_muse_release_geometry_and_output_contract_requires_explicit_acknowledgement",
            validation_status: "unvalidated",
            image_token_status: "unvalidated_text_only_fit",
        },
    },
    Profile {
        id: ProfileId::BrittLewisMuseGlimmer30bJ,
        name: MATCHED_J_PROFILE_NAME,
        method: "J",
        rule_contract: "jlens.jacobian.ordinary_autograd.v1",
        target_layer: 50,
        source_repository: MATCHED_J_SOURCE_REPOSITORY,
        source_revision: MATCHED_J_SOURCE_REVISION,
        source_filename: MATCHED_J_SOURCE_FILENAME,
        source_bytes: MATCHED_J_SOURCE_BYTES,
        source_sha256: MATCHED_J_SOURCE_SHA256,
        source_license: "no_separate_license_declared_private_research_asset",
        archive: ArchiveClaims {
            root: MATCHED_J_ARCHIVE_ROOT,
            layout: ArchiveLayout::LayerStorages,
            data_pickle_sha256: MATCHED_J_DATA_PICKLE_SHA256,
            serialization_id: Some(MATCHED_J_SERIALIZATION_ID),
            identity_layer_index: Some(50),
        },
        expected_payload_blake3: MATCHED_J_PAYLOAD_BLAKE3,
        matrix_blake3: &MATCHED_J_MATRIX_BLAKE3,
        model: ModelClaims {
            base_model: "meta-models/Muse-Glimmer-30B",
            fitted_checkpoint: "meta-models/Muse-Glimmer-30B",
            fitted_checkpoint_revision: "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
            tokenizer_checkpoint: Some("meta-models/Muse-Glimmer-30B"),
            tokenizer_revision: Some("a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
            output_rmsnorm_epsilon: 1e-5,
            output_multiplier: 0.19611613513818404,
            final_logit_softcap: 20.0,
        },
        fit: FitClaims {
            claims_basis: "pinned_source_and_opaque_data_pickle_sha256_with_audited_embedded_provenance",
            embedded_provenance: true,
            fitter: "brittlewis12/jacobian-lens/jlens",
            fitter_revision: "934ab205d0f130fb8d6f5c1224adea6fd42bffdb",
            transformers_revision: "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c",
            recipe_id: Some(
                "blank-bhatia-nanda.muse_glimmer_30b.j_lens.pile10k25.penultimate.skip4.t128.v1",
            ),
            dataset: "NeelNanda/pile-10k",
            dataset_revision: Some("127bfedcd5047750df5ccf3a12979a47bfa0bafa"),
            split: "train",
            corpus_preparation: "raw_text_add_bos_right_truncate_to_128_tokens",
            corpus_selection: Some("dataset_order_rows_0_through_24_unfiltered_unshuffled"),
            corpus_text_sha256: Some(
                "c026d7b8d3382f740a34cb3f00339ac16dd4854a81cf5eb19c7f604ee96f8632",
            ),
            corpus_token_ids_sha256: Some(
                "86146f01f323971a9bde07767b3f2e6bda241be3bc36c095d8e90f15d1c4734e",
            ),
            estimator_contract: Some(
                "jlens.causal_all_valid_targets.mean_valid_source_positions.exclude_final_position.v2",
            ),
            arithmetic_contract: Some(
                "jlens.model_dtype_forward_and_cotangent.fp32_cpu_rows_and_accumulator.v1",
            ),
            n_prompts: 25,
            max_sequence_length: 128,
            skip_first: 4,
            valid_positions_per_prompt: None,
            dim_batch: 4,
            model_execution_dtype: "bfloat16",
            serialized_dtype: "float16",
            stop_rule: "fixed_25_prompt_recipe_require_all_prompts_no_early_stop",
            convergence_status: "complete_fixed_recipe_not_convergence_measured",
            modality: "text_only",
        },
        transfer: TransferClaims {
            binding: "published_checkpoint_geometry_transfer",
            deployed_checkpoint_policy: "supported_muse_release_geometry_and_output_contract_requires_explicit_acknowledgement",
            validation_status: "unvalidated",
            image_token_status: "unvalidated_text_only_fit",
        },
    },
    Profile {
        id: ProfileId::BrittLewisMuseGlimmer30bR,
        name: R_PROFILE_NAME,
        method: "R",
        rule_contract: "jlens.relp.muse_glimmer.residual_branch_rms_detached_scale.swiglu_identity_half.attention_jacobian.v1",
        target_layer: 50,
        source_repository: R_SOURCE_REPOSITORY,
        source_revision: R_SOURCE_REVISION,
        source_filename: R_SOURCE_FILENAME,
        source_bytes: R_SOURCE_BYTES,
        source_sha256: R_SOURCE_SHA256,
        source_license: "no_separate_license_declared_private_research_asset",
        archive: ArchiveClaims {
            root: R_ARCHIVE_ROOT,
            layout: ArchiveLayout::LayerStorages,
            data_pickle_sha256: R_DATA_PICKLE_SHA256,
            serialization_id: Some(R_SERIALIZATION_ID),
            identity_layer_index: Some(50),
        },
        expected_payload_blake3: R_PAYLOAD_BLAKE3,
        matrix_blake3: &R_MATRIX_BLAKE3,
        model: ModelClaims {
            base_model: "meta-models/Muse-Glimmer-30B",
            fitted_checkpoint: "meta-models/Muse-Glimmer-30B",
            fitted_checkpoint_revision: "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
            tokenizer_checkpoint: Some("meta-models/Muse-Glimmer-30B"),
            tokenizer_revision: Some("a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
            output_rmsnorm_epsilon: 1e-5,
            output_multiplier: 0.19611613513818404,
            final_logit_softcap: 20.0,
        },
        fit: FitClaims {
            claims_basis: "pinned_source_and_opaque_data_pickle_sha256_with_audited_embedded_provenance",
            embedded_provenance: true,
            fitter: "anthropics/jacobian-lens/jlens",
            fitter_revision: "6ca15db1bfe166ea2261da24bae3c0e83e5d8495",
            transformers_revision: "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c",
            recipe_id: Some(
                "blank-bhatia-nanda.muse_glimmer_30b.r_lens.pile10k25.penultimate.skip4.t128.v1",
            ),
            dataset: "NeelNanda/pile-10k",
            dataset_revision: Some("127bfedcd5047750df5ccf3a12979a47bfa0bafa"),
            split: "train",
            corpus_preparation: "raw_text_add_bos_right_truncate_to_128_tokens",
            corpus_selection: Some("dataset_order_rows_0_through_24_unfiltered_unshuffled"),
            corpus_text_sha256: Some(
                "c026d7b8d3382f740a34cb3f00339ac16dd4854a81cf5eb19c7f604ee96f8632",
            ),
            corpus_token_ids_sha256: Some(
                "86146f01f323971a9bde07767b3f2e6bda241be3bc36c095d8e90f15d1c4734e",
            ),
            estimator_contract: Some(
                "jlens.causal_all_valid_targets.mean_valid_source_positions.exclude_final_position.v2",
            ),
            arithmetic_contract: Some(
                "jlens.model_dtype_forward_and_cotangent.fp32_cpu_rows_and_accumulator.v1",
            ),
            n_prompts: 25,
            max_sequence_length: 128,
            skip_first: 4,
            valid_positions_per_prompt: None,
            dim_batch: 4,
            model_execution_dtype: "bfloat16",
            serialized_dtype: "float16",
            stop_rule: "fixed_25_prompt_recipe_require_all_prompts_no_early_stop",
            convergence_status: "complete_fixed_recipe_not_convergence_measured",
            modality: "text_only",
        },
        transfer: TransferClaims {
            binding: "published_checkpoint_geometry_transfer",
            deployed_checkpoint_policy: "supported_muse_release_geometry_and_output_contract_requires_explicit_acknowledgement",
            validation_status: "unvalidated",
            image_token_status: "unvalidated_text_only_fit",
        },
    },
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    pub(crate) status: String,
    pub(crate) profile: String,
    pub(crate) transport: Transport,
    pub(crate) model: Model,
    pub(crate) fit: Fit,
    pub(crate) source: Source,
    pub(crate) payload: Payload,
    pub(crate) transfer: Transfer,
    pub(crate) provenance: Provenance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Transport {
    pub(crate) method: String,
    pub(crate) rule_contract: String,
    pub(crate) target_layer: u32,
    pub(crate) source_layers: Vec<u32>,
    pub(crate) coordinate: String,
    pub(crate) orientation: String,
    pub(crate) storage_dtype: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Model {
    pub(crate) architecture: String,
    pub(crate) geometry: muse_lens_artifact::Geometry,
    pub(crate) base_model: String,
    pub(crate) fitted_checkpoint: String,
    pub(crate) fitted_checkpoint_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tokenizer_checkpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tokenizer_revision: Option<String>,
    pub(crate) output_rmsnorm_epsilon: f64,
    pub(crate) output_multiplier: f64,
    pub(crate) final_logit_softcap: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Fit {
    pub(crate) claims_basis: String,
    pub(crate) embedded_provenance: bool,
    pub(crate) fitter: String,
    pub(crate) fitter_revision: String,
    pub(crate) transformers_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) recipe_id: Option<String>,
    pub(crate) dataset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) dataset_revision: Option<String>,
    pub(crate) split: String,
    pub(crate) corpus_preparation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) corpus_selection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) corpus_text_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) corpus_token_ids_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) estimator_contract: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) arithmetic_contract: Option<String>,
    pub(crate) n_prompts: u64,
    pub(crate) max_sequence_length: u32,
    pub(crate) skip_first: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) valid_positions_per_prompt: Option<u32>,
    pub(crate) dim_batch: u32,
    pub(crate) model_execution_dtype: String,
    pub(crate) serialized_dtype: String,
    pub(crate) stop_rule: String,
    pub(crate) convergence_status: String,
    pub(crate) modality: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Source {
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) filename: String,
    pub(crate) byte_length: u64,
    pub(crate) sha256: String,
    pub(crate) data_pickle_sha256: String,
    pub(crate) serialization_id: String,
    pub(crate) license: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub(crate) path: String,
    pub(crate) dtype: String,
    pub(crate) shape: [usize; 3],
    pub(crate) byte_length: u64,
    pub(crate) blake3: String,
    pub(crate) matrices: Vec<MatrixDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Transfer {
    pub(crate) binding: String,
    pub(crate) deployed_checkpoint_policy: String,
    pub(crate) validation_status: String,
    pub(crate) image_token_status: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provenance {
    pub(crate) build_commit: String,
    pub(crate) build_dirty: String,
    pub(crate) build_source_state: String,
    pub(crate) build_stamp_source: String,
    pub(crate) build_stamp_error: String,
    pub(crate) pickle_execution: String,
}

pub(crate) fn profile_for_source(byte_length: u64, sha256: &str) -> Option<Profile> {
    PROFILES.iter().copied().find(|profile| {
        valid_profile_definition(*profile)
            && profile.source_bytes() == byte_length
            && profile.source_sha256() == sha256
    })
}

pub(crate) fn profile_for_manifest(manifest: &Manifest) -> Result<Profile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| {
            valid_profile_definition(*profile)
                && manifest.profile == profile.name()
                && manifest.source.repository == profile.source_repository()
                && manifest.source.revision == profile.source_revision()
                && manifest.source.filename == profile.source_filename()
                && manifest.source.byte_length == profile.source_bytes()
                && manifest.source.sha256 == profile.source_sha256()
        })
        .context("Muse published manifest does not identify a supported pinned profile")
}

impl Profile {
    pub(crate) const fn name(self) -> &'static str {
        self.name
    }

    pub(crate) const fn source_repository(self) -> &'static str {
        self.source_repository
    }

    pub(crate) const fn source_revision(self) -> &'static str {
        self.source_revision
    }

    pub(crate) const fn source_filename(self) -> &'static str {
        self.source_filename
    }

    pub(crate) const fn source_bytes(self) -> u64 {
        self.source_bytes
    }

    pub(crate) const fn source_sha256(self) -> &'static str {
        self.source_sha256
    }

    pub(crate) const fn expected_payload_blake3(self) -> &'static str {
        self.expected_payload_blake3
    }

    pub(crate) const fn archive_spec(self) -> ArchiveSpec<'static> {
        ArchiveSpec {
            root: self.archive.root,
            layout: self.archive.layout,
            layer_count: SOURCE_LAYER_COUNT,
            hidden_size: HIDDEN_SIZE,
            matrix_bytes: MATRIX_BYTES,
            data_pickle_sha256: self.archive.data_pickle_sha256,
            serialization_id: self.archive.serialization_id,
            identity_layer_index: self.archive.identity_layer_index,
        }
    }

    pub(crate) const fn identity_layer_index(self) -> Option<usize> {
        self.archive.identity_layer_index
    }
}

fn valid_profile_definition(profile: Profile) -> bool {
    let reference = MuseGlimmerConfig::release_reference();
    let embedded_fit_complete = !profile.fit.embedded_provenance
        || profile.fit.recipe_id.is_some_and(|value| !value.is_empty())
            && profile
                .fit
                .dataset_revision
                .is_some_and(|value| !value.is_empty())
            && profile
                .fit
                .corpus_selection
                .is_some_and(|value| !value.is_empty())
            && profile.fit.corpus_text_sha256.is_some_and(is_sha256)
            && profile.fit.corpus_token_ids_sha256.is_some_and(is_sha256)
            && profile
                .fit
                .estimator_contract
                .is_some_and(|value| !value.is_empty())
            && profile
                .fit
                .arithmetic_contract
                .is_some_and(|value| !value.is_empty());
    let embedded_model_complete = !profile.fit.embedded_provenance
        || profile
            .model
            .tokenizer_checkpoint
            .is_some_and(|value| !value.is_empty())
            && profile
                .model
                .tokenizer_revision
                .is_some_and(|value| !value.is_empty());
    matches!(profile.method, "J" | "R")
        && !profile.name.is_empty()
        && !profile.rule_contract.is_empty()
        && profile.target_layer < reference.layer_count
        && profile.source_bytes >= PAYLOAD_BYTES
        && is_sha256(profile.source_sha256)
        && is_sha256(profile.archive.data_pickle_sha256)
        && is_sha256(profile.expected_payload_blake3)
        && profile.matrix_blake3.len() == SOURCE_LAYER_COUNT
        && profile.matrix_blake3.iter().all(|digest| is_sha256(digest))
        && profile
            .archive
            .serialization_id
            .is_some_and(|value| !value.is_empty())
        && profile
            .archive
            .identity_layer_index
            .is_none_or(|layer| layer < SOURCE_LAYER_COUNT)
        && (profile.method != "R"
            || profile.fit.embedded_provenance
                && profile.target_layer as usize + 1 == SOURCE_LAYER_COUNT
                && profile.archive.identity_layer_index == Some(profile.target_layer as usize)
                && profile.matrix_blake3[profile.target_layer as usize] == IDENTITY_MATRIX_BLAKE3)
        && [
            profile.source_repository,
            profile.source_revision,
            profile.source_filename,
            profile.source_license,
            profile.archive.root,
            profile.model.base_model,
            profile.model.fitted_checkpoint,
            profile.model.fitted_checkpoint_revision,
            profile.fit.claims_basis,
            profile.fit.fitter,
            profile.fit.fitter_revision,
            profile.fit.transformers_revision,
            profile.fit.dataset,
            profile.fit.split,
            profile.fit.corpus_preparation,
            profile.fit.model_execution_dtype,
            profile.fit.serialized_dtype,
            profile.fit.stop_rule,
            profile.fit.convergence_status,
            profile.fit.modality,
            profile.transfer.binding,
            profile.transfer.deployed_checkpoint_policy,
            profile.transfer.validation_status,
            profile.transfer.image_token_status,
        ]
        .iter()
        .all(|value| !value.is_empty())
        && profile.fit.n_prompts > 0
        && profile.fit.max_sequence_length > profile.fit.skip_first
        && profile
            .fit
            .valid_positions_per_prompt
            .is_none_or(|positions| positions > 0)
        && profile.fit.dim_batch > 0
        && embedded_fit_complete
        && embedded_model_complete
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn payload_from_extracted(
    profile: Profile,
    extracted: ExtractedPayload,
) -> Result<Payload> {
    ensure!(
        extracted.byte_length == PAYLOAD_BYTES
            && extracted.blake3 == profile.expected_payload_blake3()
            && extracted.matrices.len() == SOURCE_LAYER_COUNT,
        "imported Muse published payload does not match the pinned profile"
    );
    let matrices = extracted
        .matrices
        .into_iter()
        .enumerate()
        .map(|(slot, matrix)| {
            ensure!(
                matrix.archive_storage_index
                    == profile.archive.layout.storage_index_for_layer(slot)
                    && matrix.byte_offset == slot as u64 * MATRIX_BYTES
                    && matrix.byte_length == MATRIX_BYTES
                    && matrix.blake3 == profile.matrix_blake3[slot],
                "imported Muse published matrix {slot} does not match the pinned profile"
            );
            Ok(MatrixDescriptor {
                source_layer: slot as u32,
                byte_offset: matrix.byte_offset,
                byte_length: matrix.byte_length,
                blake3: matrix.blake3,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Payload {
        path: PAYLOAD_NAME.into(),
        dtype: "f16_le".into(),
        shape: [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE],
        byte_length: extracted.byte_length,
        blake3: extracted.blake3,
        matrices,
    })
}

pub(crate) fn canonical_manifest(profile: Profile, payload: Payload) -> Manifest {
    Manifest {
        schema: SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        status: "complete".into(),
        profile: profile.name().into(),
        transport: canonical_transport(profile),
        model: canonical_model(profile),
        fit: canonical_fit(profile),
        source: canonical_source(profile),
        payload,
        transfer: canonical_transfer(profile),
        provenance: Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
            pickle_execution: "none_fixed_schema_pinned_zip_entries_only".into(),
        },
    }
}

pub(crate) fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.schema == SCHEMA
            && manifest.schema_version == SCHEMA_VERSION
            && manifest.status == "complete",
        "artifact is not a complete Muse published full transport"
    );
    let profile = profile_for_manifest(manifest)?;
    ensure!(
        manifest.transport == canonical_transport(profile)
            && manifest.model == canonical_model(profile)
            && manifest.fit == canonical_fit(profile)
            && manifest.source == canonical_source(profile)
            && manifest.transfer == canonical_transfer(profile),
        "Muse published full-transport claims are not canonical"
    );
    validate_payload(profile, &manifest.payload)?;
    super::validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && matches!(manifest.provenance.build_dirty.as_str(), "0" | "1")
            && !manifest.provenance.build_stamp_source.is_empty()
            && manifest.provenance.pickle_execution == "none_fixed_schema_pinned_zip_entries_only",
        "Muse published import provenance is incomplete or permits pickle execution"
    );
    Ok(())
}

fn validate_payload(profile: Profile, payload: &Payload) -> Result<()> {
    ensure!(
        payload.path == PAYLOAD_NAME
            && payload.dtype == "f16_le"
            && payload.shape == [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE]
            && payload.byte_length == PAYLOAD_BYTES
            && payload.blake3 == profile.expected_payload_blake3
            && payload.matrices.len() == SOURCE_LAYER_COUNT,
        "Muse published payload descriptor is not canonical"
    );
    for (slot, matrix) in payload.matrices.iter().enumerate() {
        ensure!(
            matrix.source_layer == slot as u32
                && matrix.byte_offset == slot as u64 * MATRIX_BYTES
                && matrix.byte_length == MATRIX_BYTES
                && matrix.blake3 == profile.matrix_blake3[slot],
            "Muse published matrix descriptor {slot} is not canonical"
        );
    }
    Ok(())
}

fn canonical_transport(profile: Profile) -> Transport {
    Transport {
        method: profile.method.into(),
        rule_contract: profile.rule_contract.into(),
        target_layer: profile.target_layer,
        source_layers: (0..SOURCE_LAYER_COUNT as u32).collect(),
        coordinate: muse_lens_artifact::COORDINATE.into(),
        orientation: "source_layer_target_output_coordinate_source_coordinate".into(),
        storage_dtype: "f16_le".into(),
    }
}

fn canonical_model(profile: Profile) -> Model {
    let reference = MuseGlimmerConfig::release_reference();
    Model {
        architecture: ARCHITECTURE_NAME.into(),
        geometry: muse_lens_artifact::geometry(&reference),
        base_model: profile.model.base_model.into(),
        fitted_checkpoint: profile.model.fitted_checkpoint.into(),
        fitted_checkpoint_revision: profile.model.fitted_checkpoint_revision.into(),
        tokenizer_checkpoint: profile.model.tokenizer_checkpoint.map(str::to_owned),
        tokenizer_revision: profile.model.tokenizer_revision.map(str::to_owned),
        output_rmsnorm_epsilon: profile.model.output_rmsnorm_epsilon,
        output_multiplier: profile.model.output_multiplier,
        final_logit_softcap: profile.model.final_logit_softcap,
    }
}

fn canonical_fit(profile: Profile) -> Fit {
    Fit {
        claims_basis: profile.fit.claims_basis.into(),
        embedded_provenance: profile.fit.embedded_provenance,
        fitter: profile.fit.fitter.into(),
        fitter_revision: profile.fit.fitter_revision.into(),
        transformers_revision: profile.fit.transformers_revision.into(),
        recipe_id: profile.fit.recipe_id.map(str::to_owned),
        dataset: profile.fit.dataset.into(),
        dataset_revision: profile.fit.dataset_revision.map(str::to_owned),
        split: profile.fit.split.into(),
        corpus_preparation: profile.fit.corpus_preparation.into(),
        corpus_selection: profile.fit.corpus_selection.map(str::to_owned),
        corpus_text_sha256: profile.fit.corpus_text_sha256.map(str::to_owned),
        corpus_token_ids_sha256: profile.fit.corpus_token_ids_sha256.map(str::to_owned),
        estimator_contract: profile.fit.estimator_contract.map(str::to_owned),
        arithmetic_contract: profile.fit.arithmetic_contract.map(str::to_owned),
        n_prompts: profile.fit.n_prompts,
        max_sequence_length: profile.fit.max_sequence_length,
        skip_first: profile.fit.skip_first,
        valid_positions_per_prompt: profile.fit.valid_positions_per_prompt,
        dim_batch: profile.fit.dim_batch,
        model_execution_dtype: profile.fit.model_execution_dtype.into(),
        serialized_dtype: profile.fit.serialized_dtype.into(),
        stop_rule: profile.fit.stop_rule.into(),
        convergence_status: profile.fit.convergence_status.into(),
        modality: profile.fit.modality.into(),
    }
}

fn canonical_source(profile: Profile) -> Source {
    Source {
        repository: profile.source_repository.into(),
        revision: profile.source_revision.into(),
        filename: profile.source_filename.into(),
        byte_length: profile.source_bytes,
        sha256: profile.source_sha256.into(),
        data_pickle_sha256: profile.archive.data_pickle_sha256.into(),
        serialization_id: profile
            .archive
            .serialization_id
            .expect("active Muse profile requires serialization ID")
            .into(),
        license: profile.source_license.into(),
    }
}

fn canonical_transfer(profile: Profile) -> Transfer {
    Transfer {
        binding: profile.transfer.binding.into(),
        deployed_checkpoint_policy: profile.transfer.deployed_checkpoint_policy.into(),
        validation_status: profile.transfer.validation_status.into(),
        image_token_status: profile.transfer.image_token_status.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(id: ProfileId) -> Profile {
        PROFILES
            .iter()
            .copied()
            .find(|profile| profile.id == id)
            .unwrap()
    }

    fn canonical_payload(profile: Profile) -> Payload {
        Payload {
            path: PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE],
            byte_length: PAYLOAD_BYTES,
            blake3: profile.expected_payload_blake3.into(),
            matrices: profile
                .matrix_blake3
                .iter()
                .enumerate()
                .map(|(slot, digest)| MatrixDescriptor {
                    source_layer: slot as u32,
                    byte_offset: slot as u64 * MATRIX_BYTES,
                    byte_length: MATRIX_BYTES,
                    blake3: (*digest).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn canonical_j_profile_binds_publication_geometry_and_digests() {
        let profile = profile(ProfileId::EyesMlMuseGlimmer30bJ);
        assert!(valid_profile_definition(profile));
        let manifest = canonical_manifest(profile, canonical_payload(profile));
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.transport.method, "J");
        assert_eq!(manifest.transport.target_layer, 51);
        assert_eq!(
            manifest.transport.source_layers,
            (0..51).collect::<Vec<_>>()
        );
        assert_eq!(manifest.model.geometry.hidden_size, 6_656);
        assert!(manifest.model.tokenizer_revision.is_none());
        assert_eq!(manifest.fit.n_prompts, 900);

        let mut changed = manifest.clone();
        changed.payload.matrices[17].blake3 = "00".repeat(32);
        assert!(validate_manifest(&changed).is_err());
    }

    #[test]
    fn canonical_matched_j_profile_binds_recipe_identity_and_payload() {
        let matched_j = profile(ProfileId::BrittLewisMuseGlimmer30bJ);
        assert!(valid_profile_definition(matched_j));
        assert_eq!(
            profile_for_source(MATCHED_J_SOURCE_BYTES, MATCHED_J_SOURCE_SHA256)
                .unwrap()
                .id,
            ProfileId::BrittLewisMuseGlimmer30bJ
        );
        assert_eq!(matched_j.archive_spec().root, MATCHED_J_ARCHIVE_ROOT);
        assert_eq!(matched_j.archive_spec().identity_layer_index, Some(50));

        let manifest = canonical_manifest(matched_j, canonical_payload(matched_j));
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.profile, MATCHED_J_PROFILE_NAME);
        assert_eq!(manifest.transport.method, "J");
        assert_eq!(
            manifest.transport.rule_contract,
            "jlens.jacobian.ordinary_autograd.v1"
        );
        assert_eq!(manifest.transport.target_layer, 50);
        assert_eq!(
            manifest.transport.source_layers,
            (0..51).collect::<Vec<_>>()
        );
        assert_eq!(manifest.payload.blake3, MATCHED_J_PAYLOAD_BLAKE3);
        assert_eq!(manifest.payload.matrices[50].blake3, IDENTITY_MATRIX_BLAKE3);
        assert_eq!(
            manifest.model.fitted_checkpoint_revision,
            "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
        );
        assert_eq!(manifest.fit.n_prompts, 25);
        assert_eq!(manifest.fit.skip_first, 4);
        assert_eq!(manifest.fit.dim_batch, 4);
        assert!(manifest.fit.embedded_provenance);
        assert_eq!(matched_j.fit.fitter, "brittlewis12/jacobian-lens/jlens");
        assert_eq!(
            profile(ProfileId::BrittLewisMuseGlimmer30bR).fit.fitter,
            "anthropics/jacobian-lens/jlens"
        );
        assert_eq!(
            manifest.fit.recipe_id.as_deref(),
            Some("blank-bhatia-nanda.muse_glimmer_30b.j_lens.pile10k25.penultimate.skip4.t128.v1")
        );
    }

    #[test]
    fn canonical_r_profile_binds_recipe_identity_and_payload() {
        let profile = profile(ProfileId::BrittLewisMuseGlimmer30bR);
        assert!(valid_profile_definition(profile));
        assert_eq!(
            profile_for_source(R_SOURCE_BYTES, R_SOURCE_SHA256)
                .unwrap()
                .id,
            ProfileId::BrittLewisMuseGlimmer30bR
        );
        assert_eq!(profile.archive_spec().root, R_ARCHIVE_ROOT);
        assert_eq!(profile.archive_spec().identity_layer_index, Some(50));

        let manifest = canonical_manifest(profile, canonical_payload(profile));
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.profile, R_PROFILE_NAME);
        assert_eq!(manifest.transport.method, "R");
        assert_eq!(manifest.transport.target_layer, 50);
        assert_eq!(
            manifest.transport.source_layers,
            (0..51).collect::<Vec<_>>()
        );
        assert_eq!(manifest.payload.blake3, R_PAYLOAD_BLAKE3);
        assert_eq!(manifest.payload.matrices[50].blake3, IDENTITY_MATRIX_BLAKE3);
        assert_eq!(
            manifest.model.fitted_checkpoint_revision,
            "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
        );
        assert_eq!(manifest.fit.n_prompts, 25);
        assert_eq!(manifest.fit.skip_first, 4);
        assert_eq!(manifest.fit.dim_batch, 4);
        assert!(manifest.fit.embedded_provenance);
        assert_eq!(
            manifest.fit.estimator_contract.as_deref(),
            Some(
                "jlens.causal_all_valid_targets.mean_valid_source_positions.exclude_final_position.v2"
            )
        );
    }

    #[test]
    fn embedded_r_profile_cannot_activate_without_complete_recipe_provenance() {
        let mut candidate = profile(ProfileId::BrittLewisMuseGlimmer30bR);
        assert!(valid_profile_definition(candidate));

        candidate.fit.arithmetic_contract = None;
        assert!(!valid_profile_definition(candidate));

        let mut unembedded = profile(ProfileId::BrittLewisMuseGlimmer30bR);
        unembedded.fit.embedded_provenance = false;
        assert!(!valid_profile_definition(unembedded));

        let mut missing_tokenizer = profile(ProfileId::BrittLewisMuseGlimmer30bR);
        missing_tokenizer.model.tokenizer_revision = None;
        assert!(!valid_profile_definition(missing_tokenizer));

        let mut wrong_target = profile(ProfileId::BrittLewisMuseGlimmer30bR);
        wrong_target.target_layer = 49;
        wrong_target.archive.identity_layer_index = Some(49);
        assert!(!valid_profile_definition(wrong_target));

        let mut wrong_identity_digest = profile(ProfileId::BrittLewisMuseGlimmer30bR);
        wrong_identity_digest.matrix_blake3 = &MATRIX_BLAKE3;
        assert!(!valid_profile_definition(wrong_identity_digest));
    }

    #[test]
    fn extracted_payload_requires_every_pinned_matrix_digest() {
        let profile = profile(ProfileId::EyesMlMuseGlimmer30bJ);
        let extracted = ExtractedPayload {
            byte_length: PAYLOAD_BYTES,
            blake3: PAYLOAD_BLAKE3.into(),
            matrices: MATRIX_BLAKE3
                .iter()
                .enumerate()
                .map(
                    |(slot, digest)| super::super::published_pt::ExtractedMatrix {
                        archive_storage_index: slot,
                        byte_offset: slot as u64 * MATRIX_BYTES,
                        byte_length: MATRIX_BYTES,
                        blake3: (*digest).into(),
                    },
                )
                .collect(),
        };
        let payload = payload_from_extracted(profile, extracted.clone()).unwrap();
        assert_eq!(payload.blake3, PAYLOAD_BLAKE3);

        let mut changed = extracted;
        changed.matrices[50].blake3 = "11".repeat(32);
        assert!(payload_from_extracted(profile, changed).is_err());
    }
}
