
require('@g-js-api/g.js');
const fs = require('fs')

const hd = fs.readFileSync('hidden_layer_data.json', 'utf8')
const od = fs.readFileSync('output_layer_data.json', 'utf8')
const hiddenLayer = JSON.parse(hd)
const outputLayer = JSON.parse(od)

extract($)

exportConfig({
    type: 'savefile',
    options: { level_name: "test123" }
}).then(() => {

    let x = 0,
        y = 530,
        drawing = counter(),
        counters = []

    on(event(69, group(0), group(1)), trigger_function(() => {
        drawing.set(1)
    }))

    on(event(70, group(0), group(1)), trigger_function(() => {
        drawing.set(0)

    }))

    for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {

            const blockID = unknown_b(),
                groupID = unknown_g(),
                counterID = counter()

            counters.push(counterID)

            add(object({ //collision block
                OBJ_ID: 1816,
                X: x,
                Y: y,
                SCALING: 0.5,
                DYNAMIC_BLOCK: false,
                BLOCK_A: blockID,
            }))

            add(object({ //fill in block
                OBJ_ID: 2943,
                X: x,
                Y: y,
                SCALING: 0.52,
                GROUPS: groupID,
                OPACITY: 0,
            }))

            add(object({
                OBJ_ID: 1007, //alpha trigger (init block to transparent)
                FADE_IN: 0,
                OPACITY: 0,
                TARGET: groupID,
                X: x * -4 + 20 - 200,
                Y: y * 2,
                SCALING: 0.5
            }))

            on(collision(block(0), blockID), trigger_function(() => {

                drawing.if_is(EQUAL_TO, 1, trigger_function(() => {

                    groupID.alpha()
                    counterID.set(1)


                }))
            }))

            x += 15

        }
        x = 0
        y -= 15
    }

    let hidden = [counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter()],
        output = [counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter()],
        yy = 525

    for(let k = 0; k < 10; k++) {
        output[k].display(496, yy)
        yy+=30
    }

    on(collision(block(0), block(789)), trigger_function(() => {

        /* HIDDEN LAYER */

        for(let i = 0; i < 10; i++) {
            for(let j = 0; j < 784; j++) {
                const temp = counter()
                counters[j].copy_to(temp)
                temp.multiply(hiddenLayer[i].weights[j])
                hidden[i].add(temp)
            }
            hidden[i].add(hiddenLayer[i].bias)
            hidden[i].if_is(SMALLER_THAN, 0, trigger_function(() => {
                hidden[i].set(0)
            }))
        }

        /* OUTPUT LAYER */
        for(let i = 0; i < 10; i++) {
            for(let j = 0; j < 10; j++) {
                const temp = counter()
                hidden[i].copy_to(temp)
                temp.multiply(outputLayer[i].weights[j])
                output[i].add(temp)
            }
            output[i].add(outputLayer[i].bias)
            output[i].divide(10000)
        }


    }))


})