let nn;
let training_data = [
  {
    inputs:[0,0],
    outputs: [0]
  },
  {
    inputs:[0,1],
    outputs: [1]
  },
  {
    inputs:[1,0],
    outputs: [1]
  },
  {
    inputs:[1,1],
    outputs: [0]
  },
];

function setup(){
  createCanvas(400,400);
  nn = new NeuralNetwork(2,3,1);
}

function draw(){
  background(0);
  for(let i = 0; i<1000; i++){
  let data = random(training_data);
  nn.train(data.inputs,data.outputs);
  }

  let resolution = 10;
  let cols = width/ resolution;
  let rows = height/resolution;
  for(let i = 0; i< cols; i++){
    for(let j = 0; j<rows; j++){
      let x1 = i/cols;
      let x2 = j/rows;
      let inputs = [x1,x2];
      let y = nn.feedforward(inputs);
      noStroke();
      fill(y*255);
      rect(i*resolution, j*resolution, resolution, resolution)
    }
  }
}
