#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <semaphore.h>
#define NUM 5

int queue[NUM];
sem_t blank_number, product_number, mutex1;
void *producer(void *arg)
{
    int i = 0;
    while (1) {
        sem_wait(&blank_number);
        sem_wait(&mutex1);
        queue[i] = rand() % 1000 + 1; 
        printf("----Produce---%d\n", queue[i]);
        sem_post(&mutex1);
        sem_post(&product_number);
        i = (i+1) % NUM;
        sleep(rand() % 3);
    }
}

void *consumer(void *arg)
{
    int i = 0;
    while (1) {
        sem_wait(&product_number); 
        sem_wait(&mutex1);
        printf("---Consume---%d\n", queue[i]);
        queue[i] = 0; 
        sem_post(&mutex1);
        sem_post(&blank_number);
        i = (i+1) % NUM;
        sleep(rand() % 3);
    }
}

int main(int argc, char *argv[])
{
    pthread_t pid, cid;
    sem_init(&blank_number, 0, NUM);
    sem_init(&product_number, 0, 0);
    sem_init(&mutex1, 0, 1);
    pthread_create(&pid, NULL, producer, NULL);
    pthread_create(&cid, NULL, consumer, NULL);
    pthread_join(pid, NULL);
    pthread_join(cid, NULL);
    sem_destroy(&blank_number);
    sem_destroy(&product_number);
    return 0;
}
