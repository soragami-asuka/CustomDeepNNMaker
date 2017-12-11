//===============================================
// �œK�����[�`��
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_INITIALIZER_H__
#define __GRAVISBELL_LIBRARY_NN_INITIALIZER_H__

#ifdef Initializer_EXPORTS
#define Initializer_API __declspec(dllexport)
#else
#define Initializer_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#endif
#endif

#include"Layer/NeuralNetwork/IInitializer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �������Ǘ��N���X */
	class IInitializerManager
	{
	public:
		/** �R���X�g���N�^ */
		IInitializerManager(){};
		/** �f�X�g���N�^ */
		virtual ~IInitializerManager(){}

	public:
		/** ���������������� */
		virtual ErrorCode InitializeRandomParameter() = 0;
		/** ���������������� */
		virtual ErrorCode InitializeRandomParameter(U32 i_seed) = 0;

		/** �������N���X���擾���� */
		virtual IInitializer& GetInitializer(const wchar_t i_initializerID[]) = 0;
	};

	/** �������Ǘ��N���X���擾���� */
	Initializer_API IInitializerManager& GetInitializerManager(void);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
