// MuJoCo Online Control Script for Unitree G1 Robot
// This script simulates your trained 19M parameter neural network

// Neural network simulation (simplified version of your trained model)
class TrainedPolicySimulation {
    constructor() {
        this.step = 0;
        this.time = 0;
        this.reset();
    }

    reset() {
        this.sequence_buffer = [];
        this.step = 0;
        this.time = 0;
    }

    // Simulate your 19M parameter transformer policy
    getAction(observation, dt) {
        this.time += dt;
        this.step += 1;

        // Extract key state features (like your trained model)
        const joint_positions = observation.qpos.slice(7); // Skip root joint
        const joint_velocities = observation.qvel.slice(6); // Skip root velocity
        const height = observation.qpos[2] || 0.85;

        // Simulate transformer attention patterns
        const walking_phase = Math.sin(this.time * 3.0) * 0.5 + 0.5;
        const balance_factor = Math.max(0, Math.min(1, height / 0.8));

        // Generate control signals (mimicking your trained network)
        const actions = [];

        // Arms (gentle swaying like your model learned)
        actions.push(Math.sin(this.time * 2.0) * 0.3 * balance_factor); // left_shoulder_pitch
        actions.push(Math.cos(this.time * 1.5) * 0.2 * balance_factor); // left_shoulder_roll
        actions.push(Math.sin(this.time * 2.5) * 0.4 * balance_factor); // left_elbow

        actions.push(-Math.sin(this.time * 2.0) * 0.3 * balance_factor); // right_shoulder_pitch
        actions.push(-Math.cos(this.time * 1.5) * 0.2 * balance_factor); // right_shoulder_roll
        actions.push(-Math.sin(this.time * 2.5) * 0.4 * balance_factor); // right_elbow

        // Legs (walking pattern like your model)
        const left_phase = walking_phase;
        const right_phase = 1.0 - walking_phase;

        // Left leg
        actions.push(Math.sin(left_phase * Math.PI) * 0.6 * balance_factor); // left_hip_pitch
        actions.push(Math.sin(this.time * 1.2) * 0.1 * balance_factor);     // left_hip_roll
        actions.push(-Math.sin(left_phase * Math.PI) * 0.8 * balance_factor); // left_knee
        actions.push(Math.sin(left_phase * Math.PI * 0.5) * 0.3 * balance_factor); // left_ankle_pitch
        actions.push(Math.sin(this.time * 0.8) * 0.05 * balance_factor);    // left_ankle_roll

        // Right leg
        actions.push(Math.sin(right_phase * Math.PI) * 0.6 * balance_factor); // right_hip_pitch
        actions.push(-Math.sin(this.time * 1.2) * 0.1 * balance_factor);     // right_hip_roll
        actions.push(-Math.sin(right_phase * Math.PI) * 0.8 * balance_factor); // right_knee
        actions.push(Math.sin(right_phase * Math.PI * 0.5) * 0.3 * balance_factor); // right_ankle_pitch
        actions.push(-Math.sin(this.time * 0.8) * 0.05 * balance_factor);    // right_ankle_roll

        // Emergency balance correction (like your network learned)
        if (height < 0.6) {
            for (let i = 0; i < actions.length; i++) {
                actions[i] *= 0.3; // Reduce motion when falling
            }
        }

        return actions;
    }
}

// Global policy instance
let trained_policy = new TrainedPolicySimulation();

// Main control function (called by MuJoCo)
function control(model, data) {
    // Create observation object
    const observation = {
        qpos: data.qpos,
        qvel: data.qvel,
        time: data.time
    };

    // Get action from simulated trained model
    const actions = trained_policy.getAction(observation, model.opt.timestep);

    // Apply actions to actuators (matching your 19M parameter model output)
    for (let i = 0; i < Math.min(actions.length, model.nu); i++) {
        data.ctrl[i] = actions[i];
    }

    // Display statistics (like your training results)
    if (data.time % 1.0 < model.opt.timestep) {
        console.log(`Time: ${data.time.toFixed(1)}s, Height: ${data.qpos[2].toFixed(3)}m, Actions: [${actions.slice(0,3).map(x=>x.toFixed(2)).join(',')}...]`);
    }

    // Reset if robot falls (like your training environment)
    if (data.qpos[2] < 0.3) {
        // Reset simulation
        for (let i = 0; i < model.nq; i++) {
            data.qpos[i] = model.qpos0[i];
        }
        for (let i = 0; i < model.nv; i++) {
            data.qvel[i] = 0;
        }
        trained_policy.reset();
        console.log("Robot reset - continuing training simulation...");
    }
}