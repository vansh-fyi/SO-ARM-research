-- Run with dofile('/absolute/path/to/diagnostics/capture_coppelia_reference.lua')
-- in CoppeliaSim's Lua console. Read-only scene capture; no object changes.
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local directory = scriptPath:match('^(.*)/[^/]+$')
assert(sim.getSimulationState() == sim.simulation_stopped, 'Stop simulation first')
local root = sim.getObject('/so101')
local objects = {}
for _, handle in ipairs(sim.getObjectsInTree(root, sim.handle_all, 0)) do
    local kind = sim.getObjectType(handle)
    local item = {
        handle=handle, name=sim.getObjectAlias(handle),
        parent=sim.getObjectParent(handle), type=kind,
        matrix_world=sim.getObjectMatrix(handle, sim.handle_world),
        matrix_root=sim.getObjectMatrix(handle, root),
        matrix_parent=sim.getObjectMatrix(handle, sim.handle_parent),
    }
    if kind == sim.sceneobject_joint then
        item.position = sim.getJointPosition(handle)
        local cyclic, interval = sim.getJointInterval(handle)
        item.interval = {cyclic, interval}
        item.joint_type = sim.getJointType(handle)
    end
    if kind == sim.sceneobject_shape then
        item.vertices, item.indices = sim.getShapeMesh(handle)
    end
    objects[#objects+1] = item
end
local output = directory .. '/outputs/coppelia_reference.cbor'
local file = assert(io.open(output, 'wb'))
file:write(require('cbor').encode({root=root, objects=objects}))
file:close()
print('Captured ' .. #objects .. ' objects to ' .. output)
