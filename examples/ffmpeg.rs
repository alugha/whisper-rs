use std::fs::File;

use ac_ffmpeg::{
    codec::{
        audio::{
            frame::{get_channel_layout, get_sample_format, Planes},
            AudioDecoder, AudioResampler,
        },
        Decoder,
    },
    format::{
        demuxer::{Demuxer, DemuxerWithStreamInfo},
        io::IO,
    },
    Error,
};
use anyhow::{anyhow, Result};
use clap::Parser;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

/// Open a given input file.
fn open_input(path: &str) -> Result<DemuxerWithStreamInfo<File>, ac_ffmpeg::Error> {
    let input = File::open(path)
        .map_err(|err| Error::new(format!("unable to open input file {}: {}", path, err)))?;

    let io = IO::from_seekable_read_stream(input);

    Demuxer::builder()
        .build(io)?
        .find_stream_info(None)
        .map_err(|(_, err)| err)
}

fn get_mono_audio_data<'a>(planes: &'a Planes) -> &'a [f32] {
    // Ensure this is mono audio
    assert!(planes.len() == 1);

    let input = planes[0].data();
    let input_ptr = input.as_ptr();
    let input_len = input.len();
    let output_len = input_len / std::mem::size_of::<f32>();

    unsafe {
        // Create a slice of `f32` with the same length and starting address as `input`
        std::slice::from_raw_parts(input_ptr as *const f32, output_len)
    }
}

fn get_params(language: &str) -> FullParams {
    // create a params object
    // note that currently the only implemented strategy is Greedy, BeamSearch is a WIP
    // n_past defaults to 0
    let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });
    // and set the language to translate to
    params.set_language(language);
    // we also explicitly disable anything that prints to stdout
    params.set_print_special(false);
    params.set_print_progress(true);
    params.set_print_realtime(true);
    params.set_print_timestamps(true);
    params
}

/// Decode all audio frames from the first audio stream and print their
/// presentation timestamps.
fn transcribe(input: &str, model_path: &str, language: &str) -> Result<()> {
    let mut demuxer = open_input(input)?;
    let (stream_index, (stream, _)) = demuxer
        .streams()
        .iter()
        .map(|stream| (stream, stream.codec_parameters()))
        .enumerate()
        .find(|(_, (_, params))| params.is_audio_codec())
        .ok_or_else(|| Error::new("no audio stream"))?;
    let codec_params = stream.codec_parameters();
    let audio_params = codec_params
        .as_audio_codec_parameters()
        .ok_or_else(|| Error::new("no audio stream"))?;
    let mut decoder = AudioDecoder::from_stream(stream)?.build()?;

    // load a context and model
    let mut ctx = WhisperContext::new(model_path)
        .map_err(|e| anyhow!("failed loading whisper model: {:?}", e))?;

    // audio data needs to have the following format:
    // - f32
    // - 16 kHz
    // - mono channel

    let mut resampler = AudioResampler::builder()
        .source_channel_layout(audio_params.channel_layout().to_owned())
        .source_sample_format(audio_params.sample_format())
        .source_sample_rate(audio_params.sample_rate())
        .target_channel_layout(get_channel_layout("mono"))
        .target_sample_format(get_sample_format("flt"))
        .target_sample_rate(16000)
        .build()?;

    let mut full_data: Vec<f32> = Vec::new();

    // process data
    while let Some(packet) = demuxer.take()? {
        if packet.stream_index() != stream_index {
            continue;
        }

        decoder.push(packet)?;

        while let Some(frame) = decoder.take()? {
            resampler.push(frame)?;
            while let Some(frame) = resampler.take()? {
                // mono audio contains exactly one plane
                let planes = frame.planes();
                let data = get_mono_audio_data(&planes);
                full_data.extend(data);
            }
        }
    }

    decoder.flush()?;
    while let Some(frame) = decoder.take()? {
        resampler.push(frame)?;
        while let Some(frame) = resampler.take()? {
            let planes = frame.planes();
            let data = get_mono_audio_data(&planes);
            full_data.extend(data);
        }
    }

    resampler.flush()?;
    while let Some(frame) = resampler.take()? {
        let planes = frame.planes();
        let data = get_mono_audio_data(&planes);
        full_data.extend(data);
    }

    ctx.full(get_params(language), &full_data[..])
        .map_err(|e| anyhow!("failed loading whisper model: {:?}", e))?;
    // fetch the results
    let num_segments = ctx.full_n_segments();
    for i in 0..num_segments {
        let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = ctx.full_get_segment_t0(i);
        let end_timestamp = ctx.full_get_segment_t1(i);
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file
    #[arg(short, long)]
    input: String,

    /// Model file
    #[arg(short, long)]
    model: String,

    /// Language code (two characters, e.g., en, de, nl, es)
    #[arg(short, long)]
    language: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    transcribe(&args.input, &args.model, &args.language)?;
    Ok(())
}
