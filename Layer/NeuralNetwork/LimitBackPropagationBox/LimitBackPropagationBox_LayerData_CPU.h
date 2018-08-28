//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __LimitBackPropagationBox_LAYERDATA_CPU_H__
#define __LimitBackPropagationBox_LAYERDATA_CPU_H__

#include"LimitBackPropagationBox_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class LimitBackPropagationBox_LayerData_CPU : public LimitBackPropagationBox_LayerData_Base
	{
		friend class LimitBackPropagationBox_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		LimitBackPropagationBox_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~LimitBackPropagationBox_LayerData_CPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif