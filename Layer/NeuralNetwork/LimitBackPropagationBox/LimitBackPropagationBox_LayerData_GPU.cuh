//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __LimitBackPropagationBox_LAYERDATA_GPU_H__
#define __LimitBackPropagationBox_LAYERDATA_GPU_H__

#include"LimitBackPropagationBox_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class LimitBackPropagationBox_LayerData_GPU : public LimitBackPropagationBox_LayerData_Base
	{
		friend class LimitBackPropagationBox_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		LimitBackPropagationBox_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~LimitBackPropagationBox_LayerData_GPU();


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