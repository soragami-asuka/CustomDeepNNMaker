//======================================
// ��݂��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __CONVOLUTION_DATA_BASE_H__
#define __CONVOLUTION_DATA_BASE_H__

#include<Layer/IO/ISingleInputLayerData.h>
#include<Layer/IO/ISingleOutputLayerData.h>
#include<Layer/NeuralNetwork/INNLayer.h>
#include<Layer/NeuralNetwork/IOptimizer.h>

#include<vector>

#include"CONVOLUTION_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Convolution_LayerData_Base : public IO::ISingleInputLayerData, public IO::ISingleOutputLayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[�f�[�^���ʗp��GUID */

		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		Convolution::LayerStructure layerStructure;	/**< ���C���[�\�� */

		Vector3D<S32> convolutionCountVec;	/**< ��݂��݉�.(�ő�ړ���) */
		U32 convolutionCount;

		IOptimizer* m_pOptimizer_neuron;	/**< �j���[�����X�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_bias;		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Convolution_LayerData_Base(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		virtual ~Convolution_LayerData_Base();


		//===========================
		// ���ʏ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U32 GetUseBufferByteCount()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const;
		

		//===========================
		// �ŗL�֐�
		//===========================
	public:


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
	public:
		/** �I�v�e�B�}�C�U�[��ύX���� */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif