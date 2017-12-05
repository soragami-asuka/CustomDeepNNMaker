//======================================
// �������֐��̃��C���[�f�[�^
//======================================
#ifndef __Activation_Discriminator_LAYERDATA_GPU_H__
#define __Activation_Discriminator_LAYERDATA_GPU_H__

#include"Activation_Discriminator_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Activation_Discriminator_LayerData_GPU : public Activation_Discriminator_LayerData_Base
	{
		friend class Activation_Discriminator_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Activation_Discriminator_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Activation_Discriminator_LayerData_GPU();


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