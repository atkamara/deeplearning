Create Table Neurons(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  neuron_id INTEGER,
  layer_id INTEGER,
  created_at  DATE NOT NULL,
  updated_at DATE NOT NULL,
  FOREIGN KEY (layer_id) REFERENCES Layers(layer_id)
);

Create Table Layers(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  layer_id INTEGER,
  created_at  DATE NOT NULL,
  updated_at DATE NOT NULL,
  type VARCHAR(50) NOT NULL
);

Create Table Weights(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  weight_id INTEGER NOT NULL,
  neuron_id INTEGER NOT NULL,
  created_at  DATE NOT NULL,
  updated_at DATE NOT NULL,
  value INTEGER,
  FOREIGN KEY (neuron_id) REFERENCES Neurons(neuron_id)
);