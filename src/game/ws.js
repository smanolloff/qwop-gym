class WS {
  static LOG = false;

  //
  // Header (uint8)
  //

  static H_REG = 0    // reg req    (**->**) payload: id (uint8)
  static H_ACK = 1    // reg ack    (**->**)
  static H_REJ = 2    // reg rej    (**->**)
  static H_CMD = 3    // cmd        (py->js) payload: cmdflags (uint8) + step (uint16) + rew (float32) + tot_rew (float32)
  static H_OBS = 4    // obs        (js->py) payload: flags (uint8) + time (float32) + distance (float32) + obs ([60]float32)
  static H_IMG = 5    // image      (js->py) payload: format (uint8) + data (binary)
  static H_LOG = 6    // log        (js->py) payload: msg (utf-8)
  static H_ERR = 7    // error      (js->py) payload: msg (utf-8)
  static H_RLD = 8    // reload     (js->srv) payload: seed (uint32)

  //
  // Data
  //

  // REG payload: id (uint8)
  static REG_JS = 0   // js client
  static REG_PY = 1   // py client

  // CMD payload: cmdflags (uint8)
  static CMD_STP = 0b00000001  // advance 1 timestep
  static CMD_K_Q = 0b00000010  // key Q
  static CMD_K_W = 0b00000100  // key W
  static CMD_K_O = 0b00001000  // key O
  static CMD_K_P = 0b00010000  // key P
  static CMD_RST = 0b00100000  // restart game
  static CMD_IMG = 0b01000000  // capture frame (returns img instead of obs)
  static CMD_DRW = 0b10000000  // draw (browser render)

  // OBS payload: flags (uint8)
  static OBS_PAS = 0b00000001  // pass (don't act on this observation)
  static OBS_END = 0b00000010  // game has ended
  static OBS_SUC = 0b00000100  // run was successful (100+m)

  // IMG payload: format (uint8)
  static IMG_JPG = 0
  static IMG_PNG = 1


  // Default header to send
  static H_DEFAULT = 0;

  // kb/mouse events
  static DOWN_Q = new KeyboardEvent("keydown",  {keyCode: 81});
  static DOWN_W = new KeyboardEvent("keydown",  {keyCode: 87});
  static DOWN_O = new KeyboardEvent("keydown",  {keyCode: 79});
  static DOWN_P = new KeyboardEvent("keydown",  {keyCode: 80});
  static UP_Q   = new KeyboardEvent("keyup",    {keyCode: 81});
  static UP_W   = new KeyboardEvent("keyup",    {keyCode: 87});
  static UP_O   = new KeyboardEvent("keyup",    {keyCode: 79});
  static UP_P   = new KeyboardEvent("keyup",    {keyCode: 80});


  constructor(fn_reset, fn_step, fn_draw, fn_keydown, fn_keyup, fn_observation, fn_update_stats) {
    this.fn_reset = fn_reset;
    this.fn_step = fn_step;
    this.fn_draw = fn_draw;
    this.fn_keydown = fn_keydown;
    this.fn_keyup = fn_keyup;
    this.fn_observation = fn_observation;
    this.fn_update_stats = fn_update_stats;
  }

  connect(port) {
    this.ws = new WebSocket(`ws://127.0.0.1:${port}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = (_event) => this.register();
    this.ws.onmessage = (event) => this.onmessage(event);

    // this.ws.onclose = (event) => {
    //   console.log("REMOTE CLOSED CONNECTION, RELOADING");
    //   window.location.reload();
    // };
  }

  onmessage(event) {
    const dv_in = new DataView(this.recv(event));
    const header = dv_in.getUint8(0);

    // Don't do anything on non-cmd requests
    if (header != WS.H_CMD)
      return (header == WS.H_ACK) ? true : console.log("Unexpected WS header: ", header);

    this.process_cmd(dv_in)
  }

  recv(event) {
    const data = event.data;
    if (WS.LOG) console.log("< ", data);
    return data;
  }

  send(dv) {
    // DEBUG
    dv.setUint8(0, dv.getUint8(0) | WS.H_DEFAULT);

    if (WS.LOG)
      console.log("> ", dv);

    this.ws.send(dv);
  }

  // REVISIT: headers are different now
  request(cmd, negate_header) {
    const dv = new DataView(new ArrayBuffer(2))

    dv.setUint8(0, WS.H_CMD);
    dv.setUint8(1, cmd);

    this.send(dv);
  }

  process_cmd(dv_in) {
    const cmd = dv_in.getUint8(1);

    try {
      (cmd & WS.CMD_RST) && this.reset();
      (cmd & WS.CMD_K_Q) ? this.fn_keydown(WS.DOWN_Q) : this.fn_keyup(WS.UP_Q);
      (cmd & WS.CMD_K_W) ? this.fn_keydown(WS.DOWN_W) : this.fn_keyup(WS.UP_W);
      (cmd & WS.CMD_K_O) ? this.fn_keydown(WS.DOWN_O) : this.fn_keyup(WS.UP_O);
      (cmd & WS.CMD_K_P) ? this.fn_keydown(WS.DOWN_P) : this.fn_keyup(WS.UP_P);
      (cmd & WS.CMD_STP) && this.fn_step();
      (cmd & WS.CMD_DRW) && this.fn_draw();
      (cmd & WS.CMD_IMG) ? this.image() : this.observe(dv_in);
    } catch (e) {
      console.log(e.stack)
      // insert a space (1 byte) be re-written as header
      const buf = new TextEncoder().encode(" " + e.stack).buffer;
      const dv_out = new DataView(buf);
      dv_out.setUint8(0, WS.H_ERR);
      this.ws.send(dv_out);
    }
  }

  reset() {
    this.fn_keyup(WS.UP_Q);
    this.fn_keyup(WS.UP_W);
    this.fn_keyup(WS.UP_O);
    this.fn_keyup(WS.UP_P);
    this.fn_reset();
  }

  image() {
    document.getElementById("window1").toBlob((blob) => {
      blob.arrayBuffer().then((buf) => {
        const ary = new Uint8Array(2 + buf.byteLength);

        ary[0] = WS.H_IMG;
        ary[1] = WS.IMG_JPG;
        // ary.set(buf, 2); // does not work
        ary.set(new Uint8Array(buf), 2);
        this.send(new DataView(ary.buffer))
      });
    }, "image/jpeg");
  }

  observe(dv_in) {
    const dv_out = this.fn_observation();
    this.fn_update_stats(dv_in, dv_out);
    this.send(dv_out);
  }

  register() {
    const ary = new Uint8Array([WS.H_REG, WS.REG_JS]);
    this.send(new DataView(ary.buffer));
  }

  log(msg) {
    // insert a space (1 byte) to be re-written as header
    const buf = new TextEncoder().encode(" " + msg).buffer;
    const dv = new DataView(buf);
    dv.setUint8(0, WS.H_LOG);
    this.ws.send(dv);
  }
};
