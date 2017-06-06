#ifdef BATCHDATANOLISTGENERATOR_EXPORTS
#define BatchDataNoListGenerator_API __declspec(dllexport)
#else
#define BatchDataNoListGenerator_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.Common.BatchDataNoListGenerator.lib")
#endif
#endif

#include"../../Common/IBatchDataNoListGenerator.h"



namespace Gravisbell {
namespace Common {

	/** �o�b�`�����f�[�^�ԍ����X�g�����N���X���쐬����. CPU���� */
	extern BatchDataNoListGenerator_API Gravisbell::Common::IBatchDataNoListGenerator* CreateBatchDataNoListGenerator();

}	// Common
}	// Gravisbell