class HumCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      const copy = new Float32Array(channel);
      this.port.postMessage(copy);
    }
    return true;
  }
}

registerProcessor("hum-capture", HumCapture);
