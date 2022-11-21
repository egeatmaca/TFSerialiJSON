const tf = require('@tensorflow/tfjs');

async function modelToJSON(model) {
    function handleSave(artifacts) {
      return artifacts;
    }

    const modelJSON = await model.save(tf.io.withSaveHandler(handleSave));

    return modelJSON;
}

async function modelFromJSON(modelJSON) {
    const model = await tf.loadLayersModel(tf.io.fromMemory(modelJSON));
    return model;
}

async function main() {
    const model1 = tf.sequential();
    model1.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model1.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    await model1.fit(tf.tensor2d([1, 2, 3, 4], [4, 1]), tf.tensor2d([1, 3, 5, 7], [4, 1]), {epochs: 10})

    const modelJSON = await modelToJSON(model1);
    console.log('modelJSON:', modelJSON, '\n');
    console.log("typeof modelJSON:", typeof modelJSON, "\n");

    const model2 = await modelFromJSON(modelJSON);

    console.log('model1:');
    model1.summary();
    console.log('\n', "model2:");
    model2.summary();
}

main();