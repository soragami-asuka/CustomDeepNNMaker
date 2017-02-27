#ifdef BATCHDATANOLISTGENERATOR_EXPORTS
#define BatchDataNoListGenerator_API __declspec(dllexport)
#else
#define BatchDataNoListGenerator_API __declspec(dllimport)
#endif

#include"IBatchDataNoListGenerator.h"


/** �o�b�`�����f�[�^�ԍ����X�g�����N���X���쐬����. CPU���� */
extern "C" BatchDataNoListGenerator_API CustomDeepNNLibrary::IBatchDataNoListGenerator* CreateBatchDataNoListGeneratorCPU();

/** �o�b�`�����f�[�^�ԍ����X�g�����N���X���쐬����. GPU���� */
extern "C" BatchDataNoListGenerator_API CustomDeepNNLibrary::IBatchDataNoListGenerator* CreateBatchDataNoListGeneratorGPU();
