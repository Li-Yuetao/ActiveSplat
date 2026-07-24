from functools import wraps
import threading
import time
import os
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


CAMERA_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)

CAMERA_BEST_EFFORT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


def service_callback_guard(callback):
    """Return the supplied ROS2 response if a service callback raises."""
    @wraps(callback)
    def guarded(owner, request, response):
        try:
            return callback(owner, request, response)
        except Exception as exc:
            node = getattr(owner, '_Visualizer__node', owner)
            node.get_logger().error(f'{callback.__name__} failed: {exc}')
            return response
    return guarded


def call_service_sync(node, client, request, timeout_sec=10.0):
    """Bounded synchronous client call for a node already in an executor."""
    logger = node.get_logger()
    stop_event = getattr(node, 'stop_requested', None)
    deadline = time.monotonic() + float(timeout_sec)
    while not client.wait_for_service(timeout_sec=0.1):
        if stop_event is not None and stop_event.is_set():
            return None
        if time.monotonic() >= deadline:
            logger.error(f"service unavailable: {client.srv_name}")
            return None
    future = client.call_async(request)
    completed = threading.Event()
    future.add_done_callback(lambda _: completed.set())
    while not completed.wait(timeout=0.1):
        if stop_event is not None and stop_event.is_set():
            future.cancel()
            return None
        if time.monotonic() >= deadline:
            future.cancel()
            logger.error(f"service future timeout: {client.srv_name}")
            return None
    exception = future.exception()
    if exception is not None:
        logger.error(f"service future exception from {client.srv_name}: {exception}")
        return None
    response = future.result()
    if response is None:
        logger.error(f"service returned no response: {client.srv_name}")
        return None
    return response


def call_service(node, client, request, timeout_sec=10.0):
    """Compatibility name for the native bounded synchronous helper."""
    return call_service_sync(node, client, request, timeout_sec)


def wait_for_message(parent_node, msg_type, topic, qos, timeout_sec=5.0):
    context = parent_node.context

    tmp_node = rclpy.create_node(
        f"wait_for_message_tmp_{os.getpid()}_{time.time_ns()}",
        context=context,
    )

    msg_box = {"msg": None}

    def callback(msg):
        msg_box["msg"] = msg

    sub = tmp_node.create_subscription(
        msg_type,
        topic,
        callback,
        qos,
    )

    start_time = time.monotonic()

    try:
        while rclpy.ok(context=context) and msg_box["msg"] is None:
            rclpy.spin_once(tmp_node, timeout_sec=0.05)

            if time.monotonic() - start_time > timeout_sec:
                return None

        return msg_box["msg"]

    finally:
        tmp_node.destroy_subscription(sub)
        tmp_node.destroy_node()
