FROM ubuntu:22.04

RUN apt update -y
RUN apt install -y git curl xz-utils

# Install Zig and Zig language server. See:
#   https://ziglang.org/download/
#   https://install.zigtools.org/
ARG ZIG_PATH=/usr/local/zig
RUN PLATFORM=`uname -m` \
    && curl https://ziglang.org/builds/zig-linux-${PLATFORM}-0.11.0-dev.4010+70c71935c.tar.xz -o /tmp/zig.tar.xz \
    && cd /tmp/ \
    && tar -xf zig.tar.xz \
    && mv /tmp/zig-linux-${PLATFORM}-0.11.0-dev.4010+70c71935c ${ZIG_PATH} \
    && rm /tmp/zig.tar.xz \
    && curl https://zig.pm/zls/downloads/${PLATFORM}-linux/bin/zls -o /usr/local/bin/zls \
    && chmod +x /usr/local/bin/zls
ENV PATH="$PATH:${ZIG_PATH}"
