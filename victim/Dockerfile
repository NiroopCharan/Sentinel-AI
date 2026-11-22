# Victim pod: a REAL network-reachable SSH service (build prompt section 3.2), plus session auditing so the
# Identity Consistency Score has real process-lineage / timing / origin data to work with.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server procps stress-ng curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /var/run/sshd
RUN useradd -m -s /bin/bash demoUser && echo "demoUser:demoPass123" | chpasswd
# DELIBERATELY WEAK (demo only): shadow-group membership lets the credential-theft scenario read /etc/shadow
# so Falco's "Read sensitive file untrusted" rule has a successful open to fire on.
RUN usermod -aG shadow demoUser
COPY sentinel-session /usr/local/bin/sentinel-session
COPY sentinel_audit_rc /etc/sentinel_audit_rc
RUN chmod 755 /usr/local/bin/sentinel-session \
    && touch /var/log/sentinel_audit.log && chmod 666 /var/log/sentinel_audit.log \
    && echo "ForceCommand /usr/local/bin/sentinel-session" >> /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
