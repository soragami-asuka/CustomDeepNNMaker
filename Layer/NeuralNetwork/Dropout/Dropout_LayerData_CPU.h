//======================================
// �������֐��̃��C���[�f�[�^
//======================================
#ifndef __Dropout_LAYERDATA_CPU_H__
#define __Dropout_LAYERDATA_CPU_H__

#include"Dropout_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Dropout_LayerData_CPU : public Dropout_LayerData_Base
	{
		friend class Dropout_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Dropout_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Dropout_LayerData_CPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif